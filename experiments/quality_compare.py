"""HiDream-O1 on the P100a at the full setting (50 steps, CFG 5) on the prompts used for the klein vs Z-Image
comparison (1024^2, seed 1234) and on edit cases from the Qwen-Image-2.1 quality set. Needs the card
(SCRIPT=../hidream-o1/quality_compare.py via experiments/z-image-turbo/run_device_check.py).
Writes device-check/quality.json and quality-*.png."""
import json
from pathlib import Path
import sys
import time
import traceback

from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "deploy"))
from hidreamo1.pipeline import HiDreamTT, close_device, dram_stats, open_device  # noqa: E402

OUT = ROOT / "device-check"
OUT.mkdir(exist_ok=True)
STAGE = ROOT.parents[0] / "flux2-klein/stage/cases.json"
FIX = ROOT.parents[0] / "reference-image/fixtures"
EDITS = ["teapot_green", "cat_blue_sofa", "cat_red_cushion", "portrait_blue_jacket", "portrait_remove_glasses",
         "lighthouse_dusk"]


def main():
    cases = json.loads(STAGE.read_text())
    quality = {c["id"]: c for c in json.loads((ROOT.parents[0] / "reference-image/quality_cases.json").read_text())["cases"]}
    report = {"runs": []}
    dev = open_device()
    try:
        t0 = time.perf_counter()
        pipe = HiDreamTT(dev)
        report.update(load_s=time.perf_counter() - t0, dram_after_load=dram_stats(dev))
        jobs = [(c["id"], c["prompt"], 1024, 1024, c["seed"], None) for c in cases if c["task_mode"] == "text_to_image"]
        # edits: the raw instruction (HiDream's own template does the rest), output at the case size, reference as uploaded
        jobs += [(cid, quality[cid]["prompt"], quality[cid]["width"], quality[cid]["height"], quality[cid]["seed"],
                  FIX / f"{quality[cid]['fixture']}.png") for cid in EDITS]
        for name, prompt, w, h, seed, ref in jobs:
            rec = {"name": name, "width": w, "height": h}
            try:
                refs = [Image.open(ref).convert("RGB")] if ref else None
                t1 = time.perf_counter()
                image, req, tm = pipe.generate(prompt, w, h, seed, refs=refs, snap=False)
                rec.update(total_s=time.perf_counter() - t1, timing_s=tm.values, size=list(image.size))
                image.save(OUT / f"quality-{name}.png")
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"[:1500]
                rec["traceback"] = traceback.format_exc()[-2500:]
            report["runs"].append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k != "traceback"}), flush=True)
            (OUT / "quality.json").write_text(json.dumps(report, indent=2))
    finally:
        (OUT / "quality.json").write_text(json.dumps(report, indent=2))
        close_device(dev)


if __name__ == "__main__":
    main()
