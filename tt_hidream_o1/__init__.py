# SPDX-License-Identifier: Apache-2.0
"""HiDream-O1-Image on one Tenstorrent Blackhole p100a.

A pixel-space generator with no VAE: Qwen3-VL-8B plus three generation parts
(`x_embedder`, `t_embedder1`, `final_layer2`). The language model runs on the card;
the vision tower, the sampler and the patch maths run on the host.
"""
from .pipeline import HiDreamTT, Timing, close_device, dram_stats, open_device

__all__ = ["HiDreamTT", "Timing", "open_device", "close_device", "dram_stats"]
__version__ = "0.1.0"
