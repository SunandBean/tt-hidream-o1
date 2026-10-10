"""Why do full-setting HiDream edits turn into noise on the P100a? Needs the card (SCRIPT=../hidream-o1/diag_edit.py
via experiments/z-image-turbo/run_device_check.py). Runs the teapot_green edit under variants and records per-call
x_pred statistics (same fields as cpu_edit_probe.py) and the image. Writes device-check/diag_edit.json."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "deploy"))
from hidreamo1 import host  # noqa: E402
from hidreamo1.pipeline import HiDreamTT, Timing, close_device, open_device  # noqa: E402

OUT = ROOT / "device-check"
OUT.mkdir(exist_ok=True)
FIX = ROOT.parents[0] / "reference-image/fixtures/teapot.png"
PROMPT = "Change only the teapot body to green."
VARIANTS = [  # name, width, height, steps, cfg, snap, reference
    ("A_edit512_cfg5_50", 512, 512, 50, 5.0, False, True),
    ("B_edit512_cfg1_50", 512, 512, 50, 1.0, False, True),
    ("D_t2i512_cfg5_50", 512, 512, 50, 5.0, False, False),
    ("C_edit_snapped_cfg5_20", 1024, 1024, 20, 5.0, True, True),
]


def run(pipe, width, height, steps, cfg, snap, use_ref):
    refs = [Image.open(FIX).convert("RGB")] if use_ref else None
    req = host.build_request(pipe.processor, PROMPT if use_ref else "A green ceramic teapot beside a blue cup on a wooden table",
                             width, height, refs=refs, guidance_scale=cfg, snap=snap)
    timing = Timing()
    brs = pipe.branches(req, timing)
    z = host.init_noise(req.width, req.height, 42000)
    sched = host.Schedule(steps, guidance_scale=cfg)
    calls = []
    try:
        for step_t in sched.timesteps:
            t_emb = pipe.time(sched.model_time(step_t))
            patches = z if req.ref_patches is None else torch_cat(z, req.ref_patches)
            xs = [pipe.lm.step(br, patches, t_emb) for br in brs]
            for x in xs:
                xf = x.float()
                calls.append({"std": float(xf.std()), "absmax": float(xf.abs().max()), "mean": float(xf.mean())})
            z = sched.step(step_t, z, xs[0], xs[1] if len(xs) > 1 else None)
    finally:
        for br in brs:
            pipe.lm.release(br)
    return host.to_image(z, req.h_patches, req.w_patches), req, calls


def torch_cat(a, b):
    import torch
    return torch.cat([a, b])


def main():
    report = {"variants": {}}
    only = set(filter(None, os.environ.get("VARIANTS", "").split(",")))
    dev = open_device()
    try:
        pipe = HiDreamTT(dev)
        for name, w, h, steps, cfg, snap, use_ref in VARIANTS:
            if only and name not in only:
                continue
            rec = {"width": w, "height": h, "steps": steps, "cfg": cfg, "snap": snap, "reference": use_ref}
            try:
                t0 = time.perf_counter()
                image, req, calls = run(pipe, w, h, steps, cfg, snap, use_ref)
                rec.update(total_s=time.perf_counter() - t0, size=list(image.size), calls=calls,
                           seq=getattr(req, "seq_len", None))
                image.save(OUT / f"diag-{name}.png")
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"[:1500]
                rec["traceback"] = traceback.format_exc()[-2500:]
            report["variants"][name] = rec
            print(name, json.dumps({k: v for k, v in rec.items() if k not in ("calls", "traceback")}),
                  "first/last calls", rec.get("calls", [])[:2], rec.get("calls", [])[-2:], flush=True)
            (OUT / "diag_edit.json").write_text(json.dumps(report, indent=2))
    finally:
        (OUT / "diag_edit.json").write_text(json.dumps(report, indent=2))
        close_device(dev)


if __name__ == "__main__":
    main()
