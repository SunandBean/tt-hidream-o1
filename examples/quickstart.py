# SPDX-License-Identifier: Apache-2.0
"""Minimal text-to-image run on one p100a.

    python examples/quickstart.py "Three red apples and two green pears on a wooden table"

HiDream-O1 is the undistilled model: 50 steps at CFG 5 means 100 forwards, about 19 s at 1024x1024.
"""
import sys

from tt_hidream_o1 import HiDreamTT, close_device, dram_stats, open_device

prompt = sys.argv[1] if len(sys.argv) > 1 else "Three red apples and two green pears on a wooden table, still life photograph"

dev = open_device()
try:
    model = HiDreamTT(dev)   # downloads the snapshot into the HF cache on first use
    print(f"device DRAM {dram_stats(dev)['allocated_bytes'] / 2**30:.1f} GiB")

    # snap=False generates the size asked for; the official pipeline would round it to ~4 MP.
    # generate() returns the image, the request it resolved, and the timings.
    image, _req, timing = model.generate(prompt, 1024, 1024, seed=1234, snap=False)
    image.save("output.png")
    print({k: round(v, 2) for k, v in timing.values.items()})
finally:
    close_device(dev)
