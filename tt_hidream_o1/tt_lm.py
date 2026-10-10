# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""HiDream-O1-Image generator (the Qwen3-VL-8B text stack + pixel patch heads) on ONE Blackhole, TTNN, batch 1.

Mirrors ref_algo.RefHiDream with one layout change the math allows: attention is order-free over keys and RoPE
positions are explicit, so the generation block is laid out [target patches; reference patches; pad | tms-tile]
where the tms tile is 32 rows (row 0 = timestep embedding, rest masked). x_embedder then runs as one matmul on
the uploaded patches and each step uploads only the patches and one 32-row tile.

Ops follow the Z-Image / Qwen-Image-2.1 single-chip ports (Apache-2.0 Tenstorrent code): fused QKV matmul with
per-head permuted q/k rows, nlp_create_qkv_heads (GQA 32/8), per-head RMSNorm, adjacent-pair
rotary_embedding_llama, SDPA (HiFi4 + fp32 accumulation, the setting that held Z-Image parity), fused SwiGLU
minimal_matmul."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import ttnn

from . import host
from .config import GEN, LM, TILE, LMConfig

MEM = ttnn.DRAM_MEMORY_CONFIG
NEG = -1e9

try:
    from models.tt_dit.utils.matmul import get_matmul_config
except ImportError:  # pragma: no cover
    get_matmul_config = None


def _as_4d(t):
    if len(t.shape) == 4:
        return t
    return ttnn.reshape(t, [1] * (4 - len(t.shape)) + list(t.shape))


def _dev(dev, t: torch.Tensor, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=MEM)


def _row(dev, v: torch.Tensor):
    return _dev(dev, v.to(torch.bfloat16).reshape(1, 1, 1, -1))


@dataclass
class LMPrecision:
    weight_dtype: ttnn.DataType = ttnn.bfloat16
    mm_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    sdpa_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4
    sdpa_fp32_acc: bool = True


class Layer:
    def __init__(self, dev, get, idx: int, prec: LMPrecision, perm, cfg: LMConfig):
        g = lambda k: get(f"model.language_model.layers.{idx}.{k}")
        wd = prec.weight_dtype
        wq = host.permute_heads_rows(g("self_attn.q_proj.weight"), cfg.heads, cfg.head_dim, perm)
        wk = host.permute_heads_rows(g("self_attn.k_proj.weight"), cfg.kv_heads, cfg.head_dim, perm)
        self.wqkv = _dev(dev, torch.cat([host.linear_to_mm(wq), host.linear_to_mm(wk),
                                         host.linear_to_mm(g("self_attn.v_proj.weight"))], 1), wd)
        self.q_norm = _row(dev, g("self_attn.q_norm.weight")[perm])
        self.k_norm = _row(dev, g("self_attn.k_norm.weight")[perm])
        self.wo = _dev(dev, host.linear_to_mm(g("self_attn.o_proj.weight")), wd)
        self.w_gateup = _dev(dev, host.swiglu_interleave(g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")), wd)
        self.w_down = _dev(dev, host.linear_to_mm(g("mlp.down_proj.weight")), wd)
        self.ln1 = _row(dev, g("input_layernorm.weight"))
        self.ln2 = _row(dev, g("post_attention_layernorm.weight"))


@dataclass
class Prefix:
    k: List[ttnn.Tensor]  # per layer [1, KVH, Tp, D]
    v: List[ttnn.Tensor]
    length: int
    padded: int


@dataclass
class Branch:
    """Per CFG branch device state for one request: prefix K/V, generation RoPE tables and key mask."""

    prefix: Prefix
    cos: ttnn.Tensor
    sin: ttnn.Tensor
    mask: ttnn.Tensor  # [1, 1, Gp, Tp + Gp]
    n_img: int  # target + reference patch tokens
    n_img_pad: int  # rows before the tms tile
    tgt_len: int

    @property
    def gen_padded(self) -> int:
        return self.n_img_pad + TILE


class HiDreamLM:
    def __init__(self, dev, get, prec: Optional[LMPrecision] = None, cfg: LMConfig = LM):
        self.dev, self.cfg = dev, cfg
        self.prec = prec or LMPrecision()
        self.perm = host.interleave_pairs_permutation(cfg.head_dim)
        self.layers = [Layer(dev, get, i, self.prec, self.perm, cfg) for i in range(cfg.num_layers)]
        self.norm = _row(dev, get("model.language_model.norm.weight"))
        self.x1 = _dev(dev, host.linear_to_mm(get("model.x_embedder.proj1.weight")))
        self.x2 = _dev(dev, host.linear_to_mm(get("model.x_embedder.proj2.weight")))
        self.x2b = _row(dev, get("model.x_embedder.proj2.bias"))
        self.fw = _dev(dev, host.linear_to_mm(get("model.final_layer2.linear.weight")))
        self.fb = _row(dev, get("model.final_layer2.linear.bias"))
        self.trans_mat = _dev(dev, host.rot_transformation_mat())
        self.grid = dev.compute_with_storage_grid_size()
        arch = dev.arch()
        ck = lambda fid, fp32: ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False)
        self.ck_mm = ck(self.prec.mm_fidelity, True)
        self.ck_norm = ck(ttnn.MathFidelity.HiFi4, True)
        self.ck_sdpa = ck(self.prec.sdpa_fidelity, self.prec.sdpa_fp32_acc)
        self._mm_cfg: Dict[tuple, object] = {}
        self.sdpa_chunks = (256, 128, 64, 32)

    # ------------------------------------------------------------------ ops
    def _mm(self, x, w, M, K, N, fuse_swiglu=False):
        key = (M, K, N)
        if key not in self._mm_cfg:
            self._mm_cfg[key] = get_matmul_config(M, K, N, self.grid)
        return _as_4d(ttnn.experimental.minimal_matmul(x, w, config=self._mm_cfg[key], compute_kernel_config=self.ck_mm,
                                                       dtype=ttnn.bfloat16, memory_config=MEM, fuse_swiglu=fuse_swiglu))

    def _rms(self, x, w):
        return ttnn.rms_norm(x, epsilon=self.cfg.rms_eps, weight=w, compute_kernel_config=self.ck_norm,
                             memory_config=MEM)

    def _chunk(self, n):
        return next(c for c in self.sdpa_chunks if n % c == 0)

    def _qkv(self, h, L: Layer, S, cos, sin):
        c = self.cfg
        qkv = self._mm(h, L.wqkv, S, c.hidden, (c.heads + 2 * c.kv_heads) * c.head_dim)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=c.heads, num_kv_heads=c.kv_heads,
                                                         transpose_k_heads=False, memory_config=MEM)
        ttnn.deallocate(qkv)
        out = []
        for t, w in ((q, L.q_norm), (k, L.k_norm)):
            n = self._rms(t, w)
            ttnn.deallocate(t)
            out.append(ttnn.experimental.rotary_embedding_llama(n, cos, sin, self.trans_mat, is_decode_mode=False,
                                                                compute_kernel_config=self.ck_norm))
            ttnn.deallocate(n)
        return out[0], out[1], v

    def _mlp_residual(self, x, L: Layer, S):
        c = self.cfg
        h = self._rms(x, L.ln2)
        m = self._mm(h, L.w_gateup, S, c.hidden, 2 * c.intermediate, fuse_swiglu=True)
        ttnn.deallocate(h)
        d = self._mm(m, L.w_down, S, c.intermediate, c.hidden)
        ttnn.deallocate(m)
        out = ttnn.add(x, d, memory_config=MEM)
        ttnn.deallocate(d)
        ttnn.deallocate(x)
        return out

    def _tables(self, pos):
        cos, sin = host.mrope_cos_sin(pos)
        f = lambda t: _dev(self.dev, t.reshape(1, 1, t.shape[0], -1))
        return f(cos), f(sin)

    # ------------------------------------------------------------------ prefix (once per prompt and CFG branch)
    def prefix(self, embeds: torch.Tensor, positions: torch.Tensor, deepstack: Optional[List[torch.Tensor]] = None
               ) -> Prefix:
        """embeds [T0, 4096] host (text embeddings, VLM image rows already substituted); deepstack: per early layer a
        dense [T0, 4096] host tensor holding the deepstack features at the vision positions (zeros elsewhere)."""
        c = self.cfg
        T0 = embeds.shape[0]
        Tp = host.round_up(T0)
        x = _dev(self.dev, host.pad_rows(embeds.to(torch.bfloat16), Tp).reshape(1, 1, Tp, -1))
        pos = torch.zeros(3, Tp, dtype=torch.long)
        pos[:, :T0] = positions
        cos, sin = self._tables(pos)
        cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=self.grid, q_chunk_size=TILE, k_chunk_size=TILE,
                                     exp_approx_mode=False)
        ks, vs = [], []
        for li, L in enumerate(self.layers):
            h = self._rms(x, L.ln1)
            q, k, v = self._qkv(h, L, Tp, cos, sin)
            ttnn.deallocate(h)
            ks.append(k)
            vs.append(v)
            a = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, scale=c.head_dim ** -0.5,
                                                              program_config=cfg, compute_kernel_config=self.ck_sdpa,
                                                              memory_config=MEM)
            ttnn.deallocate(q)
            a2 = _as_4d(ttnn.transformer.concatenate_heads(a, memory_config=MEM))
            ttnn.deallocate(a)
            o = self._mm(a2, L.wo, Tp, c.heads * c.head_dim, c.hidden)
            ttnn.deallocate(a2)
            x2 = ttnn.add(x, o, memory_config=MEM)
            ttnn.deallocate(o)
            ttnn.deallocate(x)
            x = self._mlp_residual(x2, L, Tp)
            if deepstack is not None and li < len(deepstack):
                d = _dev(self.dev, host.pad_rows(deepstack[li].to(torch.bfloat16), Tp).reshape(1, 1, Tp, -1))
                x2 = ttnn.add(x, d, memory_config=MEM)
                ttnn.deallocate(d)
                ttnn.deallocate(x)
                x = x2
        ttnn.deallocate(x)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        return Prefix(ks, vs, T0, Tp)

    def branch(self, sample: host.Sample, prefix: Prefix) -> Branch:
        pos, valid, n_img, n_pad = host.gen_layout(sample)
        cos, sin = self._tables(pos)
        Gp = n_pad + TILE
        m = torch.full((Gp, prefix.padded + Gp), NEG)
        m[:, : prefix.length] = 0.0
        m[:, prefix.padded:][:, valid] = 0.0
        mask = _dev(self.dev, m.to(torch.bfloat16).reshape(1, 1, Gp, -1))
        return Branch(prefix, cos, sin, mask, n_img, n_pad, sample.tgt_len)

    def release(self, br: Branch):
        for t in (br.cos, br.sin, br.mask, *br.prefix.k, *br.prefix.v):
            ttnn.deallocate(t)

    # ------------------------------------------------------------------ step
    def step(self, br: Branch, patches: torch.Tensor, t_emb: torch.Tensor, taps: Optional[list] = None
             ) -> torch.Tensor:
        """patches [n_img, 3072] host ([z; refs]), t_emb [4096] host -> x_pred of the target rows [tgt_len, 3072]."""
        c = self.cfg
        Gp, Tp = br.gen_padded, br.prefix.padded
        p = _dev(self.dev, host.pad_rows(patches.to(torch.bfloat16), br.n_img_pad).reshape(1, 1, br.n_img_pad, -1))
        e1 = self._mm(p, self.x1, br.n_img_pad, GEN.patch_dim, GEN.bottleneck)
        ttnn.deallocate(p)
        e2 = self._mm(e1, self.x2, br.n_img_pad, GEN.bottleneck, c.hidden)
        ttnn.deallocate(e1)
        xe = ttnn.add(e2, self.x2b, memory_config=MEM)
        ttnn.deallocate(e2)
        tile = torch.zeros(TILE, c.hidden, dtype=torch.bfloat16)
        tile[0] = t_emb.to(torch.bfloat16)
        tt_tile = _dev(self.dev, tile.reshape(1, 1, TILE, -1))
        x = ttnn.concat([xe, tt_tile], dim=2, memory_config=MEM)
        for t in (xe, tt_tile):
            ttnn.deallocate(t)
        cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=self.grid, q_chunk_size=self._chunk(Gp),
                                     k_chunk_size=self._chunk(Tp + Gp), exp_approx_mode=False)
        for L, pk, pv in zip(self.layers, br.prefix.k, br.prefix.v):
            h = self._rms(x, L.ln1)
            q, k, v = self._qkv(h, L, Gp, br.cos, br.sin)
            ttnn.deallocate(h)
            kk = ttnn.concat([pk, k], dim=2, memory_config=MEM)
            vv = ttnn.concat([pv, v], dim=2, memory_config=MEM)
            for t in (k, v):
                ttnn.deallocate(t)
            a = ttnn.transformer.scaled_dot_product_attention(q, kk, vv, attn_mask=br.mask, is_causal=False,
                                                              scale=c.head_dim ** -0.5, program_config=cfg,
                                                              compute_kernel_config=self.ck_sdpa, memory_config=MEM)
            for t in (q, kk, vv):
                ttnn.deallocate(t)
            a2 = _as_4d(ttnn.transformer.concatenate_heads(a, memory_config=MEM))
            ttnn.deallocate(a)
            o = self._mm(a2, L.wo, Gp, c.heads * c.head_dim, c.hidden)
            ttnn.deallocate(a2)
            x2 = ttnn.add(x, o, memory_config=MEM)
            ttnn.deallocate(o)
            ttnn.deallocate(x)
            x = self._mlp_residual(x2, L, Gp)
            if taps is not None:
                taps.append(ttnn.to_torch(x)[0, 0].float())
        xs = ttnn.slice(x, [0, 0, 0, 0], [1, 1, host.round_up(br.tgt_len), c.hidden], memory_config=MEM)
        ttnn.deallocate(x)
        xn = self._rms(xs, self.norm)
        ttnn.deallocate(xs)
        out = self._mm(xn, self.fw, host.round_up(br.tgt_len), c.hidden, GEN.patch_dim)
        ttnn.deallocate(xn)
        o2 = ttnn.add(out, self.fb, memory_config=MEM)
        ttnn.deallocate(out)
        host_out = ttnn.to_torch(o2)[0, 0, : br.tgt_len].float()
        ttnn.deallocate(o2)
        return host_out
