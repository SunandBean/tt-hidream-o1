# SPDX-License-Identifier: Apache-2.0
"""HiDream-O1-Image on one P100a: text/image prefix once per prompt, UniPC + CFG loop on host, generation tokens on
the device every step. No VAE: the final latents are the image patches."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from PIL import Image

from . import host
from .config import GEN, snapshot_dir


def open_device(trace_region_size: int = 0, l1_small_size: int = 32768):
    import ttnn

    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1),
                                 dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
                                 trace_region_size=trace_region_size, l1_small_size=l1_small_size)


def close_device(dev):
    import ttnn

    ttnn.close_mesh_device(dev)


def dram_stats(dev) -> Dict[str, int]:
    import ttnn

    ttnn.synchronize_device(dev)
    v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    banks = int(v.num_banks)
    return {"total_bytes": int(v.total_bytes_per_bank) * banks,
            "allocated_bytes": int(v.total_bytes_allocated_per_bank) * banks,
            "free_bytes": int(v.total_bytes_free_per_bank) * banks}


@dataclass
class Timing:
    values: Dict[str, float] = field(default_factory=dict)

    def mark(self, key: str, started: float):
        self.values[key] = self.values.get(key, 0.0) + time.perf_counter() - started


class HiDreamTT:
    EMBED = "model.language_model.embed_tokens.weight"

    def __init__(self, dev, snapshot: Optional[str] = None, prec=None):
        from transformers import AutoProcessor

        from .tt_lm import HiDreamLM

        self.dev = dev
        self.snapshot = snapshot or snapshot_dir()
        t0 = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(self.snapshot)
        self.ckpt = host.LazyCheckpoint(self.snapshot)
        get = lambda k: self.ckpt.get(k)
        self.lm = HiDreamLM(dev, get, prec)
        self.time = host.TimeEmbedder(get)
        self._vision = None
        self.load_s = time.perf_counter() - t0

    @property
    def vision(self):
        if self._vision is None:
            self._vision = host.Vision(self.snapshot, self.ckpt)
        return self._vision

    def branches(self, req: host.Request, timing: Timing):
        vision = None
        if req.samples[0].pixel_values is not None:
            t0 = time.perf_counter()
            vision = self.vision(req.samples[0])  # same reference pixels for every CFG branch
            timing.mark("vision_s", t0)
        out = []
        t0 = time.perf_counter()
        for s in req.samples:
            rows = self.ckpt.get_rows(self.EMBED, s.input_ids[: s.prefix_len])
            emb, deepstack = host.prefix_inputs(rows, s, vision)
            pre = self.lm.prefix(emb, s.position_ids[:, : s.prefix_len], deepstack)
            out.append(self.lm.branch(s, pre))
        timing.mark("prefix_s", t0)
        return out

    def denoise(self, req: host.Request, seed: int, steps: int = GEN.steps, guidance_scale: float = GEN.guidance_scale,
                timing: Optional[Timing] = None, first: Optional[list] = None) -> torch.Tensor:
        timing = timing or Timing()
        brs = self.branches(req, timing)
        z = host.init_noise(req.width, req.height, seed)
        sched = host.Schedule(steps, guidance_scale=guidance_scale)
        try:
            for step_t in sched.timesteps:
                t_emb = self.time(sched.model_time(step_t))
                patches = z if req.ref_patches is None else torch.cat([z, req.ref_patches])
                t0 = time.perf_counter()
                xs = [self.lm.step(br, patches, t_emb) for br in brs]
                timing.mark("lm_s", t0)
                if first is not None and not first:
                    first.extend(xs)
                z = sched.step(step_t, z, xs[0], xs[1] if len(xs) > 1 else None)
        finally:
            for br in brs:
                self.lm.release(br)
        return z

    def generate(self, prompt: str, width: int, height: int, seed: int, refs: Optional[List[Image.Image]] = None,
                 steps: int = GEN.steps, guidance_scale: float = GEN.guidance_scale, snap: bool = True,
                 keep_original_aspect: bool = False):
        timing = Timing()
        req = host.build_request(self.processor, prompt, width, height, refs=refs, guidance_scale=guidance_scale,
                                 keep_original_aspect=keep_original_aspect, snap=snap)
        z = self.denoise(req, seed, steps, guidance_scale, timing)
        return host.to_image(z, req.h_patches, req.w_patches), req, timing
