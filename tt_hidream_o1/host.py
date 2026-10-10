# SPDX-License-Identifier: Apache-2.0
"""Host-side (CPU) parts of the single-P100a HiDream-O1-Image port.

Mirrors the official pipeline (upstream models/pipeline.py generate_image, batch 1) and is checked against it
in experiments/hidream-o1/tests: sample building (chat template + <|boi|><|tms|>, placeholder vision tokens only
for the 3-D mRoPE positions, token types), pixel patchify, noise, UniPC schedule + CFG, interleaved mRoPE tables
in the adjacent-pair layout of rotary_embedding_llama, and the timestep embedding.

One deliberate option: snap=False skips find_closest_resolution, which in the official code forces every
text-to-image request onto ~4-5 MP sizes (2048x2048, 2560x1440, ...)."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

from .config import GEN, LM, TILE
from .upstream.utils import calculate_dimensions, find_closest_resolution, get_rope_index_fix_point, resize_pilimage

TMS_TOKEN = "<|tms_token|>"
BOI_TOKEN = "<|boi_token|>"


def round_up(n: int, m: int = TILE) -> int:
    return (n + m - 1) // m * m


# ----------------------------------------------------------------------------- checkpoint access
class LazyCheckpoint:
    def __init__(self, root: str):
        idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.key_to_file = {k: os.path.join(root, v) for k, v in idx.items()}
        self._handles: Dict[str, object] = {}

    def _handle(self, path):
        if path not in self._handles:
            self._handles[path] = safe_open(path, framework="pt", device="cpu")
        return self._handles[path]

    def keys(self):
        return self.key_to_file.keys()

    def get(self, key: str, dtype: Optional[torch.dtype] = torch.bfloat16) -> torch.Tensor:
        t = self._handle(self.key_to_file[key]).get_tensor(key)
        return t.to(dtype) if dtype is not None and t.dtype != dtype else t.clone()

    def get_rows(self, key: str, rows: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
        sl = self._handle(self.key_to_file[key]).get_slice(key)
        return torch.stack([sl[i: i + 1][0] for i in rows.reshape(-1).tolist()], dim=0).to(dtype)


# ----------------------------------------------------------------------------- weight layout helpers
def linear_to_mm(w: torch.Tensor) -> torch.Tensor:
    return w.t().contiguous()


def interleave_pairs_permutation(head_dim: int) -> torch.Tensor:
    """new[j] = old[p[j]]: rotate_half pairs (i, i + D/2) -> adjacent pairs (2i, 2i + 1)."""
    half = head_dim // 2
    p = torch.empty(head_dim, dtype=torch.long)
    p[0::2] = torch.arange(half)
    p[1::2] = torch.arange(half) + half
    return p


def permute_heads_rows(w_out_in: torch.Tensor, n_heads: int, head_dim: int, perm: torch.Tensor) -> torch.Tensor:
    out, inp = w_out_in.shape
    assert out == n_heads * head_dim
    return w_out_in.view(n_heads, head_dim, inp)[:, perm, :].reshape(out, inp).contiguous()


def swiglu_interleave(gate_out_in: torch.Tensor, up_out_in: torch.Tensor, tile: int = TILE) -> torch.Tensor:
    g, u = linear_to_mm(gate_out_in), linear_to_mm(up_out_in)
    K, N = g.shape
    return torch.stack([g.view(K, N // tile, tile), u.view(K, N // tile, tile)], dim=2).reshape(K, 2 * N).contiguous()


def rot_transformation_mat(tile: int = TILE) -> torch.Tensor:
    m = torch.zeros(1, 1, tile, tile)
    m[..., torch.arange(0, tile, 2), torch.arange(1, tile, 2)] = 1.0
    m[..., torch.arange(1, tile, 2), torch.arange(0, tile, 2)] = -1.0
    return m


def pad_rows(t: torch.Tensor, rows: int) -> torch.Tensor:
    if t.shape[-2] == rows:
        return t
    out = torch.zeros(*t.shape[:-2], rows, t.shape[-1], dtype=t.dtype)
    out[..., : t.shape[-2], :] = t
    return out


# ----------------------------------------------------------------------------- pixels
def pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    """The pipeline's TENSOR_TRANSFORM (torchvision v2 ToImage / ToDtype(scale) / Normalize 0.5): uint8 RGB ->
    float32 [3, H, W] in [-1, 1]. torchvision itself is used: a hand-written /255 differs by up to 3e-5."""
    import torchvision.transforms.v2 as T

    tf = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize([0.5], [0.5])])
    return tf(pil.convert("RGB"))


def patchify(x: torch.Tensor, p: int = GEN.patch) -> torch.Tensor:
    """[C, H, W] -> [(H/p)(W/p), C*p*p]  (einops 'C (H p1) (W p2) -> (H W) (C p1 p2)')."""
    c, h, w = x.shape
    return x.reshape(c, h // p, p, w // p, p).permute(1, 3, 0, 2, 4).reshape((h // p) * (w // p), c * p * p)


def unpatchify(z: torch.Tensor, h_patches: int, w_patches: int, p: int = GEN.patch) -> torch.Tensor:
    c = z.shape[-1] // (p * p)
    return z.reshape(h_patches, w_patches, c, p, p).permute(2, 0, 3, 1, 4).reshape(c, h_patches * p, w_patches * p)


def to_image(z: torch.Tensor, h_patches: int, w_patches: int) -> Image.Image:
    img = unpatchify((z.float() + 1) / 2, h_patches, w_patches)
    arr = np.round(np.clip(img.numpy().transpose(1, 2, 0) * 255, 0, 255)).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def init_noise(width: int, height: int, seed: int) -> torch.Tensor:
    noise = GEN.noise_scale * torch.randn((1, 3, height, width), generator=torch.Generator("cpu").manual_seed(seed + 1))
    return patchify(noise.to(torch.bfloat16)[0])


# ----------------------------------------------------------------------------- samples
@dataclass
class Sample:
    """One CFG branch: text (with tms as its last token) followed by target / reference image tokens."""

    input_ids: torch.Tensor  # [T] text incl. the tms token
    position_ids: torch.Tensor  # [3, S] for the whole sequence
    tgt_len: int
    ref_lens: List[int] = field(default_factory=list)
    pixel_values: Optional[torch.Tensor] = None  # VLM-side reference pixels (processor output)
    image_grid_thw: Optional[torch.Tensor] = None

    @property
    def txt_len(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def prefix_len(self) -> int:  # causal (AR) text tokens; the tms token is a generation token
        return self.txt_len - 1

    @property
    def gen_len(self) -> int:
        return 1 + self.tgt_len + sum(self.ref_lens)

    @property
    def vision_mask(self) -> torch.Tensor:  # image placeholders inside the prefix (deepstack positions)
        return self.input_ids[: self.prefix_len] == LM.image_token_id


@dataclass
class Request:
    samples: List[Sample]  # [cond] or [cond, uncond]
    width: int
    height: int
    ref_patches: Optional[torch.Tensor] = None  # [R, 3072] bf16

    @property
    def h_patches(self) -> int:
        return self.height // GEN.patch

    @property
    def w_patches(self) -> int:
        return self.width // GEN.patch


def _chat(processor, content):
    return processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True)


def build_request(processor, prompt: str, width: int, height: int, refs: Optional[List[Image.Image]] = None,
                  guidance_scale: float = GEN.guidance_scale, keep_original_aspect: bool = False,
                  snap: bool = True) -> Request:
    """generate_image's sample construction (official models/pipeline.py), layout conditioning excluded."""
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    P = GEN.patch
    refs = refs or []
    pre = None
    if keep_original_aspect and len(refs) == 1:
        pre = resize_pilimage(refs[0].convert("RGB"), 2048, P)
        width, height = pre.size
    elif snap:
        width, height = find_closest_resolution(width, height)
    tgt_len = (height // P) * (width // P)
    captions = [prompt] + ([" "] if guidance_scale > 1.0 else [])
    suffix = BOI_TOKEN + TMS_TOKEN

    if not refs:
        samples = []
        for caption in captions:
            ids = tok.encode(_chat(processor, caption) + suffix, return_tensors="pt", add_special_tokens=False)
            vision = torch.full((1, tgt_len), LM.image_token_id, dtype=ids.dtype)
            vision[0, 0] = LM.vision_start_token_id
            pos, _ = get_rope_index_fix_point(
                1, LM.image_token_id, LM.video_token_id, LM.vision_start_token_id,
                input_ids=torch.cat([ids, vision], -1),
                image_grid_thw=torch.tensor([[1, height // P, width // P]]), video_grid_thw=None,
                attention_mask=None, skip_vision_start_token=[1])
            samples.append(Sample(ids[0], pos[:, 0], tgt_len))
        return Request(samples, width, height)

    K = len(refs)
    m = max(height, width)
    max_size = m if K == 1 else m * 48 // 64 if K == 2 else m // 2 if K <= 4 else m * 24 // 64 if K <= 8 else m // 4
    resized = [pre if pre is not None else resize_pilimage(r.convert("RGB"), max_size, P) for r in refs]
    ref_patches = torch.cat([patchify(pil_to_tensor(r)) for r in resized]).to(torch.bfloat16)
    ref_lens = [(r.height // P) * (r.width // P) for r in resized]
    cond_size = GEN.condition_image_size if K <= 4 else GEN.condition_image_size * 48 // 64 if K <= 8 \
        else GEN.condition_image_size // 2
    vlm_pils = []
    for r in resized:
        cw, ch = calculate_dimensions(cond_size, r.width / r.height)
        vlm_pils.append(r.resize((cw, ch), resample=Image.LANCZOS))
    grid_tgt = torch.tensor([[1, height // P, width // P]])
    grid_ref = torch.tensor([[1, r.height // P, r.width // P] for r in resized])
    samples = []
    for caption in captions:
        content = [{"type": "image"} for _ in range(K)] + [{"type": "text", "text": caption}]
        proc = processor(text=[_chat(processor, content)], images=vlm_pils, padding="longest", return_tensors="pt")
        ids = torch.cat([proc.input_ids, tok.encode(suffix, return_tensors="pt", add_special_tokens=False)], -1)
        grid_cond = proc.image_grid_thw.clone()
        grid_cond[:, 1:] //= 2  # spatial_merge_size
        vision = []
        for n in [tgt_len] + ref_lens:
            v = torch.full((1, n), LM.image_token_id, dtype=ids.dtype)
            v[0, 0] = LM.vision_start_token_id
            vision.append(v)
        pos, _ = get_rope_index_fix_point(
            1, LM.image_token_id, LM.video_token_id, LM.vision_start_token_id,
            input_ids=torch.cat([ids] + vision, -1), image_grid_thw=torch.cat([grid_cond, grid_tgt, grid_ref]),
            video_grid_thw=None, attention_mask=None, skip_vision_start_token=[0] * K + [1] + [1] * K)
        samples.append(Sample(ids[0], pos[:, 0], tgt_len, ref_lens, proc.pixel_values.to(torch.bfloat16),
                              proc.image_grid_thw))
    return Request(samples, width, height, ref_patches)


def gen_layout(sample: Sample):
    """(positions [3, Gp], key-valid mask over the generation block [Gp], n_img, n_img_pad) for the reordered
    block [target; references; pad | tms; pad]."""
    T0, n_img = sample.prefix_len, sample.tgt_len + sum(sample.ref_lens)
    n_pad = round_up(n_img)
    Gp = n_pad + TILE
    pos = torch.zeros(3, Gp, dtype=torch.long)
    pos[:, :n_img] = sample.position_ids[:, T0 + 1: T0 + 1 + n_img]
    pos[:, n_pad] = sample.position_ids[:, T0]
    valid = torch.zeros(Gp, dtype=torch.bool)
    valid[:n_img] = True
    valid[n_pad] = True
    return pos, valid, n_img, n_pad


# ----------------------------------------------------------------------------- RoPE
def _inv_freq() -> torch.Tensor:
    d = LM.head_dim
    return 1.0 / (LM.rope_theta ** (torch.arange(0, d, 2, dtype=torch.int64).float() / d))


def mrope_freqs(pos: torch.Tensor) -> torch.Tensor:
    """[3, S] positions -> [S, 64] float32 interleaved-mRoPE angles (Qwen3VLTextRotaryEmbedding)."""
    freqs = pos[:, :, None].float() * _inv_freq()[None, None, :]
    out = freqs[0].clone()
    for dim, offset in ((1, 1), (2, 2)):
        length = LM.mrope_section[dim] * 3
        out[..., offset:length:3] = freqs[dim][..., offset:length:3]
    return out


def mrope_cos_sin(pos: torch.Tensor, dtype=torch.bfloat16):
    """cos/sin [S, 128] in the adjacent-pair layout (q/k rows permuted by interleave_pairs_permutation)."""
    f = mrope_freqs(pos)
    return (f.cos().to(dtype).repeat_interleave(2, -1), f.sin().to(dtype).repeat_interleave(2, -1))


def apply_rope_adjacent(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
    return x * cos + rot * sin


# ----------------------------------------------------------------------------- vision (references only)
class Vision:
    """The Qwen3-VL vision tower (27 blocks + merger + 3 deepstack mergers) on host, loaded from model.visual.*.
    Runs once per request with references; returns (image embeds [n, 4096], deepstack [3 x (n, 4096)])."""

    def __init__(self, snapshot: str, ckpt: "LazyCheckpoint"):
        from transformers import AutoConfig

        # the official (transformers-4.57-derived) vision code: transformers 5.x's own Qwen3-VL vision tower
        # returns different embeddings for the same weights and pixels
        from .upstream.qwen3_vl_transformers import Qwen3VLVisionModel

        cfg = AutoConfig.from_pretrained(snapshot).vision_config
        self.model = Qwen3VLVisionModel._from_config(cfg, torch_dtype=torch.bfloat16).eval()
        prefix = "model.visual."
        sd = {k[len(prefix):]: ckpt.get(k) for k in ckpt.keys() if k.startswith(prefix)}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.endswith("inv_freq")]
        if missing or unexpected:
            raise RuntimeError(f"vision weights mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")

    @torch.no_grad()
    def __call__(self, sample: Sample):
        out = self.model(sample.pixel_values.to(torch.bfloat16), grid_thw=sample.image_grid_thw)
        if isinstance(out, tuple):  # transformers 4.57: (merged embeds, deepstack list)
            embeds, deepstack = out
        else:  # newer transformers return a ModelOutput
            embeds = getattr(out, "pooler_output", None)
            if embeds is None:
                embeds = out.last_hidden_state
            deepstack = getattr(out, "deepstack_features", None) or getattr(out, "deepstack_feature_lists")
        return embeds, list(deepstack)


def prefix_inputs(embed_rows: torch.Tensor, sample: Sample, vision=None):
    """Prefix embeddings [T0, 4096] (VLM image rows substituted) and dense per-layer deepstack additions."""
    x = embed_rows.clone()
    if vision is None:
        return x, None
    embeds, deepstack = vision
    m = sample.vision_mask
    x[m] = embeds.to(x.dtype)
    dense = []
    for d in deepstack:
        full = torch.zeros_like(x)
        full[m] = d.to(x.dtype)
        dense.append(full)
    return x, dense


# ----------------------------------------------------------------------------- time
def timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimeEmbedder:
    """model.t_embedder1 in the checkpoint dtype (bf16), like TimestepEmbedder.forward."""

    def __init__(self, get):
        self.w0, self.b0 = get("model.t_embedder1.mlp.0.weight"), get("model.t_embedder1.mlp.0.bias")
        self.w2, self.b2 = get("model.t_embedder1.mlp.2.weight"), get("model.t_embedder1.mlp.2.bias")

    def __call__(self, t_model: float) -> torch.Tensor:
        f = timestep_embedding(torch.tensor([t_model], dtype=torch.float32) * 1000).to(self.w0.dtype)
        h = torch.nn.functional.silu(torch.nn.functional.linear(f, self.w0, self.b0))
        return torch.nn.functional.linear(h, self.w2, self.b2)[0]  # [4096]


# ----------------------------------------------------------------------------- schedule
class Schedule:
    """UniPC multistep (shift 3) + CFG exactly as generate_image's loop for the full model."""

    def __init__(self, steps: int = GEN.steps, shift: float = GEN.shift, guidance_scale: float = GEN.guidance_scale):
        from .upstream.fm_solvers_unipc import FlowUniPCMultistepScheduler

        self.sched = FlowUniPCMultistepScheduler(use_dynamic_shifting=False, shift=shift)
        self.sched.set_timesteps(steps, device="cpu")
        self.guidance_scale = guidance_scale

    @property
    def timesteps(self):
        return self.sched.timesteps

    @staticmethod
    def model_time(step_t) -> float:
        return float(1.0 - step_t.float() / 1000.0)

    def step(self, step_t, z: torch.Tensor, x_cond: torch.Tensor, x_uncond: Optional[torch.Tensor]) -> torch.Tensor:
        sigma = (step_t.float() / 1000.0).to(torch.float32).clamp_min(GEN.t_eps)
        v = (x_cond.float() - z.float()) / sigma
        if x_uncond is not None:
            vu = (x_uncond.float() - z.float()) / sigma
            v = vu + self.guidance_scale * (v - vu)
        out = self.sched.step((-v).float()[None], step_t.to(torch.float32), z.float()[None], return_dict=False)[0]
        return out[0].to(torch.bfloat16)
