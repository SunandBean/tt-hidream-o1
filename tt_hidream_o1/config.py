# SPDX-License-Identifier: Apache-2.0
"""Static configuration of HiDream-ai/HiDream-O1-Image @ 0b0901d (config.json + official inference defaults)."""
from __future__ import annotations

import os
from dataclasses import dataclass

HF_REPO = "HiDream-ai/HiDream-O1-Image"
HF_REVISION = "0b0901d99f200389e138c61946af1185f5f49a13"
UPSTREAM_CODE = "HiDream-ai/HiDream-O1-Image@2c2d29ff729e48f33e41f49edfdbd81d5ac103b4"
TILE = 32


@dataclass(frozen=True)
class LMConfig:
    """Qwen3-VL-8B text stack that is the generator itself (36 layers, all used)."""

    num_layers: int = 36
    hidden: int = 4096
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    intermediate: int = 12288
    rms_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    mrope_section: tuple = (24, 20, 20)  # interleaved T/H/W over the 64 frequencies
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    tms_token_id: int = 151673


@dataclass(frozen=True)
class GenConfig:
    patch: int = 32  # pixel patches, 3 * 32 * 32 = 3072 values per token, no VAE
    in_channels: int = 3
    bottleneck: int = 1024  # x_embedder proj1: 3072 -> hidden // 4
    noise_scale: float = 8.0  # pipeline NOISE_SCALE (full model)
    t_eps: float = 0.001
    steps: int = 50  # full (undistilled) model
    guidance_scale: float = 5.0
    shift: float = 3.0
    condition_image_size: int = 384  # VLM-side reference size (K <= 4)

    @property
    def patch_dim(self) -> int:
        return self.in_channels * self.patch * self.patch


LM = LMConfig()
GEN = GenConfig()


def snapshot_dir() -> str:
    """The local HiDream-O1-Image snapshot, downloaded into the HF cache if it is not there yet.

    HIDREAM_SNAPSHOT overrides it with a directory you manage yourself.
    """
    env = os.environ.get("HIDREAM_SNAPSHOT")
    if env:
        return env
    cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    d = os.path.join(cache, "models--" + HF_REPO.replace("/", "--"), "snapshots", HF_REVISION)
    if os.path.isdir(d):
        return d
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=HF_REPO, revision=HF_REVISION)
