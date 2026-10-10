"""P100a parity + timing check of the single-chip HiDream-O1-Image port against the CPU goldens.

Run through the card hand-off runner (stops/restores the photo service):
  SCRIPT=../hidream-o1/hidream_device_check.py python3 experiments/z-image-turbo/run_device_check.py
Writes experiments/hidream-o1/device-check/report.json and PNGs. Per golden case:
  * request rebuilt by host.build_request == golden ids / positions
  * step 0 on the golden's exact inputs: cond and uncond x_pred vs golden
  * full run with the golden's (reduced) step count: final image vs the golden image
Then (QUALITY=1, default) one full-quality 1024x1024 run: 50 UniPC steps, CFG 5. Stages are saved as they finish."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "deploy"))
from hidreamo1 import host  # noqa: E402
from hidreamo1.pipeline import HiDreamTT, Timing, close_device, dram_stats, open_device  # noqa: E402

GOLDEN = ROOT / "golden"
OUT = ROOT / "device-check"
OUT.mkdir(exist_ok=True)
report = {"status": "starting", "cases": {}}


def save():
    (OUT / "report.json").write_text(json.dumps(report, indent=2))


def pcc(a, b):
    a, b = torch.as_tensor(a).flatten().double(), torch.as_tensor(b).flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    selected = set(filter(None, os.environ.get("CASES", "").split(",")))
    dev = open_device()
    try:
        t0 = time.perf_counter()
        pipe = HiDreamTT(dev)
        report.update(load_s=time.perf_counter() - t0, dram_after_load=dram_stats(dev))
        save()
        for case_dir in sorted(p for p in GOLDEN.iterdir() if (p / "tensors.pt").exists()):
            if selected and case_dir.name not in selected:
                continue
            name = case_dir.name
            rec = json.loads((case_dir / "record.json").read_text())
            case = rec["case"]
            calls = torch.load(case_dir / "tensors.pt", weights_only=False)["calls"]
            r = report["cases"][name] = {"case": case, "seq_len": rec["seq_len"]}
            try:
                refs = [Image.open(ROOT / case["ref"])] if case.get("ref") else None
                req = host.build_request(pipe.processor, case["prompt"], case.get("width", 2048),
                                         case.get("height", 2048), refs=refs,
                                         keep_original_aspect=case.get("keep_original_aspect", False),
                                         snap=case.get("snap", True))
                r["size"] = [req.width, req.height]
                r["ids_match"] = all(torch.equal(s.input_ids, c["input_ids"][0]) for s, c in zip(req.samples, calls))
                r["positions_match"] = all(torch.equal(s.position_ids, c["position_ids"][:, 0])
                                           for s, c in zip(req.samples, calls))
                timing = Timing()
                brs = pipe.branches(req, timing)
                r["prefix_timing_s"] = dict(timing.values)
                vin = calls[0]["vinputs"][0]
                t_model = float(calls[0]["timestep"].reshape(-1)[0])
                t_emb = pipe.time(t_model)
                step0 = []
                for s, br, c in zip(req.samples, brs, calls):
                    t1 = time.perf_counter()
                    x = pipe.lm.step(br, vin, t_emb)
                    dt = time.perf_counter() - t1
                    golden = c["x_pred"][0][s.txt_len: s.txt_len + s.tgt_len]
                    step0.append({"pcc": pcc(x, golden), "max_abs": float((x - golden).abs().max()), "step_s": dt})
                t1 = time.perf_counter()
                pipe.lm.step(brs[0], vin, t_emb)  # warm repeat (programs compiled)
                r["warm_forward_s"] = time.perf_counter() - t1
                for br in brs:
                    pipe.lm.release(br)
                r["step0"] = step0
                save()
                tm = Timing()
                t1 = time.perf_counter()
                z = pipe.denoise(req, case["seed"], case["steps"], 5.0, tm)
                r["full_s"] = time.perf_counter() - t1
                r["full_timing_s"] = tm.values
                image = host.to_image(z, req.h_patches, req.w_patches)
                image.save(OUT / f"{name}.png")
                gold_img = np.asarray(Image.open(case_dir / "image.png").convert("RGB"), dtype=np.float32)
                r["image_pcc"] = pcc(np.asarray(image, dtype=np.float32), gold_img)
                r["dram"] = dram_stats(dev)
            except Exception as exc:
                r["error"] = f"{type(exc).__name__}: {exc}"[:2000]
                r["traceback"] = traceback.format_exc()[-4000:]
            print(name, json.dumps({k: v for k, v in r.items() if k not in ("case", "traceback", "dram")}), flush=True)
            save()
        if os.environ.get("QUALITY", "1") == "1":
            q = report["quality"] = {}
            for name, prompt, w, h, seed in [
                ("fox1024", "A red fox sitting in fresh snow at golden hour, wildlife photograph, sharp focus", 1024, 1024, 32),
                ("nook832x640", "A cozy reading nook with a cat sleeping on a knitted blanket, warm afternoon light",
                 832, 640, 7)]:
                try:
                    t1 = time.perf_counter()
                    image, req, tm = pipe.generate(prompt, w, h, seed, snap=False)
                    q[name] = {"total_s": time.perf_counter() - t1, "timing_s": tm.values, "size": list(image.size)}
                    image.save(OUT / f"quality-{name}.png")
                except Exception as exc:
                    q[name] = {"error": f"{type(exc).__name__}: {exc}"[:2000], "traceback": traceback.format_exc()[-4000:]}
                print(name, json.dumps({k: v for k, v in q[name].items() if k != "traceback"}), flush=True)
                save()
        report["status"] = "ok"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        save()
        close_device(dev)


if __name__ == "__main__":
    main()
