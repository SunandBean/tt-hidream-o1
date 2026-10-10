"""CPU ground truth for the diverging edits: does the OFFICIAL pipeline itself produce a valid image for a
512x512 un-snapped edit at the full setting (50 steps, CFG 5)? Records per-call x_pred statistics so the
device run can be compared step by step. No accelerator. Writes golden/edit512_probe/{image.png,stats.json}."""
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
from hidream_models import pipeline as hp  # noqa: E402
from hidream_models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

SNAPSHOT = os.environ.get("HIDREAM_SNAPSHOT",
                          "/hf/hub/models--HiDream-ai--HiDream-O1-Image/snapshots/0b0901d99f200389e138c61946af1185f5f49a13")
FIX = ROOT.parents[0] / "reference-image/fixtures/teapot.png"
PROMPT = "Change only the teapot body to green."
OUT = ROOT / "golden" / os.environ.get("PROBE_NAME", "edit512_probe")


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "12")))
    OUT.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(SNAPSHOT)
    tok = processor.tokenizer
    for name in ("boi", "bor", "eor", "bot", "tms"):
        setattr(tok, f"{name}_token", f"<|{name}_token|>")
    model = Qwen3VLForConditionalGeneration.from_pretrained(SNAPSHOT, torch_dtype=torch.bfloat16).eval()
    stats = []
    real_forward = model.forward

    def forward(*args, **kwargs):
        result = real_forward(*args, **kwargs)
        x = result.x_pred.detach().float()
        stats.append({"std": float(x.std()), "absmax": float(x.abs().max()), "mean": float(x.mean())})
        if len(stats) % 10 == 0:
            (OUT / "stats.json").write_text(json.dumps({"calls": stats}, indent=2))
        return result

    model.forward = forward
    width, height = int(os.environ.get("WIDTH", "512")), int(os.environ.get("HEIGHT", "512"))
    snap = os.environ.get("SNAP", "0") == "1"
    if not snap:
        hp.find_closest_resolution = lambda w, h: (w, h)
    started = time.perf_counter()
    image = hp.generate_image(model=model, processor=processor, prompt=PROMPT, ref_image_paths=[str(FIX)], height=height,
                              width=width, num_inference_steps=int(os.environ.get("STEPS", "50")),
                              guidance_scale=float(os.environ.get("CFG", "5.0")), shift=3.0, scheduler_name="default",
                              seed=42000, keep_original_aspect=False)
    image.save(OUT / "image.png")
    (OUT / "stats.json").write_text(json.dumps({"calls": stats, "total_s": time.perf_counter() - started,
                                                "size": list(image.size), "snap": snap}, indent=2))
    print("done", image.size, round(time.perf_counter() - started, 1), flush=True)


if __name__ == "__main__":
    main()
