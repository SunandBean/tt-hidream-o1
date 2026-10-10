"""CPU: the port's algorithm (deploy/hidreamo1 ref_algo, prefix K/V + generation tokens, both the reference order
and the device layout) on the real weights vs the golden first calls of the official pipeline (cond + uncond).
Rebuilds each request with host.build_request and checks it reproduces the golden input ids / positions first."""
import json
import os
from pathlib import Path
import sys
import time

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT.parents[1] / "deploy"))
from hidreamo1 import host, ref_algo  # noqa: E402
from hidreamo1.config import snapshot_dir  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

GOLDEN = ROOT / "golden"


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "12")))
    snap = snapshot_dir()
    processor = AutoProcessor.from_pretrained(snap)
    ckpt = host.LazyCheckpoint(snap)
    ref = ref_algo.RefHiDream(lambda k: ckpt.get(k))
    ref.attn_fp32 = os.environ.get("ATTN_FP32") == "1"
    vis = None
    selected = set(filter(None, os.environ.get("CASES", "").split(",")))
    results = {}
    for case_dir in sorted(p for p in GOLDEN.iterdir() if (p / "tensors.pt").exists()):
        if selected and case_dir.name not in selected:
            continue
        case = json.loads((case_dir / "record.json").read_text())["case"]
        calls = torch.load(case_dir / "tensors.pt", weights_only=False)["calls"]
        refs = [Image.open(ROOT / case["ref"])] if case.get("ref") else None
        req = host.build_request(processor, case["prompt"], case.get("width", 2048), case.get("height", 2048),
                                 refs=refs, keep_original_aspect=case.get("keep_original_aspect", False),
                                 snap=case.get("snap", True))
        rec = {"size": [req.width, req.height], "branches": []}
        for sample, call in zip(req.samples, calls):
            b = {"ids_match": bool(torch.equal(sample.input_ids, call["input_ids"][0])),
                 "positions_match": bool(torch.equal(sample.position_ids, call["position_ids"][:, 0]))}
            vin = call["vinputs"][0]
            z, refp = vin[: sample.tgt_len], (vin[sample.tgt_len:] if sample.ref_lens else None)
            if req.ref_patches is not None:
                b["ref_patches_match"] = bool(torch.equal(req.ref_patches, refp))
            t_model = float(call["timestep"].reshape(-1)[0])
            golden = call["x_pred"][0][sample.txt_len: sample.txt_len + sample.tgt_len]
            t0 = time.perf_counter()
            vision = None
            if sample.pixel_values is not None:
                vis = vis or host.Vision(snap, ckpt)
                vision = vis(sample)
            with torch.no_grad():
                pre = ref.prefix(sample, vision)
                ours = ref.step(sample, pre, z, refp, t_model).float()
                ours_tt = ref.step_tt_layout(sample, pre, z, refp, t_model).float()
            b.update(t_model=t_model, pcc=pcc(ours, golden), pcc_device_layout=pcc(ours_tt, golden),
                     max_abs=float((ours - golden).abs().max()), cpu_s=time.perf_counter() - t0)
            rec["branches"].append(b)
            print(case_dir.name, json.dumps(b), flush=True)
        results[case_dir.name] = rec
        (GOLDEN / ("ref_algo_check_fp32attn.json" if ref.attn_fp32 else "ref_algo_check.json")).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
