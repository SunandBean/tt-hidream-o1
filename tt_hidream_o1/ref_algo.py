# SPDX-License-Identifier: Apache-2.0
"""Torch implementation of exactly the algorithm the TT port runs, for CPU parity checks.

The official forward runs text + generation tokens in one sequence with a mixed mask (text rows causal over
text, generation rows see everything). Text rows never see generation tokens, so their per-layer K/V do not
depend on the timestep or the noisy image: the port computes them ONCE per prompt (prefix) and every step
runs only the generation tokens [tms; target image; references] against [prefix K/V; own K/V], mask-free
except for the tile padding of both parts. q/k projection rows are permuted per head so interleaved mRoPE is
applied with the adjacent-pair kernel layout."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import host
from .config import GEN, LM, LMConfig

NEG = -1e9


def rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3VLTextRMSNorm: normalize in fp32, cast back, then multiply by the (bf16) weight."""
    xf = x.float()
    y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    return w * y


@dataclass
class PrefixCache:
    k: List[torch.Tensor]  # per layer [KVH, Tp, D] (post-norm, post-RoPE)
    v: List[torch.Tensor]
    length: int  # valid prefix tokens T0 (Tp = round_up(T0))


class RefHiDream:
    def __init__(self, get: Callable[[str], torch.Tensor], cfg: LMConfig = LM, pad: bool = True):
        self.cfg, self.pad = cfg, pad
        self.perm = host.interleave_pairs_permutation(cfg.head_dim)
        self.layers = []
        for i in range(cfg.num_layers):
            p = f"model.language_model.layers.{i}."
            g = lambda k: get(p + k)
            self.layers.append({
                "wq": host.permute_heads_rows(g("self_attn.q_proj.weight"), cfg.heads, cfg.head_dim, self.perm),
                "wk": host.permute_heads_rows(g("self_attn.k_proj.weight"), cfg.kv_heads, cfg.head_dim, self.perm),
                "wv": g("self_attn.v_proj.weight"), "wo": g("self_attn.o_proj.weight"),
                "qn": g("self_attn.q_norm.weight")[self.perm], "kn": g("self_attn.k_norm.weight")[self.perm],
                "ln1": g("input_layernorm.weight"), "ln2": g("post_attention_layernorm.weight"),
                "gate": g("mlp.gate_proj.weight"), "up": g("mlp.up_proj.weight"), "down": g("mlp.down_proj.weight"),
            })
        self.norm = get("model.language_model.norm.weight")
        self.embed_w = get("model.language_model.embed_tokens.weight")
        self.x1 = get("model.x_embedder.proj1.weight")
        self.x2, self.x2b = get("model.x_embedder.proj2.weight"), get("model.x_embedder.proj2.bias")
        self.fw, self.fb = get("model.final_layer2.linear.weight"), get("model.final_layer2.linear.bias")
        self.time = host.TimeEmbedder(get)

    def _len(self, n: int) -> int:
        return host.round_up(n) if self.pad else n

    def _qkv(self, h, L, cos, sin):
        c = self.cfg
        S = h.shape[0]
        q = rms(F.linear(h, L["wq"]).view(S, c.heads, c.head_dim), L["qn"], c.rms_eps)
        k = rms(F.linear(h, L["wk"]).view(S, c.kv_heads, c.head_dim), L["kn"], c.rms_eps)
        v = F.linear(h, L["wv"]).view(S, c.kv_heads, c.head_dim)
        q = host.apply_rope_adjacent(q, cos[:, None], sin[:, None])
        k = host.apply_rope_adjacent(k, cos[:, None], sin[:, None])
        return q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)  # [H, S, D]

    def _mlp(self, x, L):
        return F.linear(F.silu(F.linear(x, L["gate"])) * F.linear(x, L["up"]), L["down"])

    attn_fp32 = False  # compute attention in fp32 (the device SDPA accumulates in fp32; CPU bf16 masked SDPA does not)

    def _attend(self, q, k, v, mask=None, causal=False):
        c = self.cfg
        rep = c.heads // c.kv_heads
        k, v = k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0)
        dt = q.dtype
        if self.attn_fp32:
            q, k, v = q.float(), k.float(), v.float()
            mask = mask.float() if mask is not None else None
        out = F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=mask, is_causal=causal,
                                             scale=c.head_dim ** -0.5)[0]
        return out.to(dt)

    def text_embeds(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed_w[ids]

    def prefix(self, sample: host.Sample, vision: Optional[Tuple[torch.Tensor, List[torch.Tensor]]] = None
               ) -> PrefixCache:
        """Causal pass over the AR text tokens (everything before tms), with vision placeholders replaced by the
        VLM image embeddings and deepstack features added after the first layers. Returns per-layer K/V."""
        c = self.cfg
        T0 = sample.prefix_len
        Tp = self._len(T0)
        x = self.text_embeds(sample.input_ids[:T0]).clone()
        vmask = sample.vision_mask
        if vision is not None:
            x[vmask] = vision[0].to(x.dtype)
        x = host.pad_rows(x, Tp)
        pos = torch.zeros(3, Tp, dtype=torch.long)
        pos[:, :T0] = sample.position_ids[:, :T0]
        cos, sin = host.mrope_cos_sin(pos)
        ks, vs = [], []
        for li, L in enumerate(self.layers):
            h = rms(x, L["ln1"], c.rms_eps)
            q, k, v = self._qkv(h, L, cos, sin)
            ks.append(k)
            vs.append(v)
            a = self._attend(q, k, v, causal=True).transpose(0, 1).reshape(Tp, -1)
            x = x + F.linear(a, L["wo"])
            x = x + self._mlp(rms(x, L["ln2"], c.rms_eps), L)
            if vision is not None and li < len(vision[1]):
                x[:T0][vmask] = x[:T0][vmask] + vision[1][li].to(x.dtype)
        return PrefixCache(ks, vs, T0)

    def gen_embeds(self, z: torch.Tensor, refs: Optional[torch.Tensor], t_model: float) -> torch.Tensor:
        vin = z if refs is None else torch.cat([z, refs])
        xe = F.linear(F.linear(vin.to(torch.bfloat16), self.x1), self.x2, self.x2b)
        return torch.cat([self.time(t_model)[None].to(xe.dtype), xe])

    def step(self, sample: host.Sample, pre: PrefixCache, z: torch.Tensor, refs: Optional[torch.Tensor],
             t_model: float, taps: Optional[list] = None) -> torch.Tensor:
        """Generation tokens only -> x_pred of the target image tokens [tgt_len, 3072]."""
        c = self.cfg
        G = sample.gen_len
        Gp = self._len(G)
        T0 = pre.length
        Tp = pre.k[0].shape[1]
        x = host.pad_rows(self.gen_embeds(z, refs, t_model), Gp)
        pos = torch.zeros(3, Gp, dtype=torch.long)
        pos[:, :G] = sample.position_ids[:, T0: T0 + G]
        cos, sin = host.mrope_cos_sin(pos)
        mask = torch.zeros(Gp, Tp + Gp)
        mask[:, T0:Tp] = NEG
        mask[:, Tp + G:] = NEG
        mask = mask.to(x.dtype)
        for li, L in enumerate(self.layers):
            h = rms(x, L["ln1"], c.rms_eps)
            q, k, v = self._qkv(h, L, cos, sin)
            kk, vv = torch.cat([pre.k[li], k], 1), torch.cat([pre.v[li], v], 1)
            a = self._attend(q, kk, vv, mask=mask).transpose(0, 1).reshape(Gp, -1)
            x = x + F.linear(a, L["wo"])
            x = x + self._mlp(rms(x, L["ln2"], c.rms_eps), L)
            if taps is not None:
                taps.append(x[:G].float().clone())
        x = rms(x, self.norm, c.rms_eps)
        out = F.linear(x, self.fw, self.fb)
        return out[1: 1 + sample.tgt_len]

    def step_tt_layout(self, sample: host.Sample, pre: PrefixCache, z: torch.Tensor, refs: Optional[torch.Tensor],
                       t_model: float) -> torch.Tensor:
        """Same step in the device layout ([target; refs; pad | tms tile], host.gen_layout) to prove the reorder."""
        c = self.cfg
        pos, valid, n_img, n_pad = host.gen_layout(sample)
        Gp = n_pad + 32
        T0, Tp = pre.length, pre.k[0].shape[1]
        vin = z if refs is None else torch.cat([z, refs])
        xe = F.linear(F.linear(host.pad_rows(vin.to(torch.bfloat16), n_pad), self.x1), self.x2, self.x2b)
        tile = torch.zeros(32, c.hidden, dtype=xe.dtype)
        tile[0] = self.time(t_model).to(xe.dtype)
        x = torch.cat([xe, tile])
        cos, sin = host.mrope_cos_sin(pos)
        mask = torch.full((Gp, Tp + Gp), NEG)
        mask[:, :T0] = 0.0
        mask[:, Tp:][:, valid] = 0.0
        mask = mask.to(x.dtype)
        for li, L in enumerate(self.layers):
            h = rms(x, L["ln1"], c.rms_eps)
            q, k, v = self._qkv(h, L, cos, sin)
            a = self._attend(q, torch.cat([pre.k[li], k], 1), torch.cat([pre.v[li], v], 1), mask=mask)
            x = x + F.linear(a.transpose(0, 1).reshape(Gp, -1), L["wo"])
            x = x + self._mlp(rms(x, L["ln2"], c.rms_eps), L)
        x = rms(x[: sample.tgt_len], self.norm, c.rms_eps)
        return F.linear(x, self.fw, self.fb)
