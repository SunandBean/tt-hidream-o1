"""CPU goldens for the single-P100a HiDream-O1-Image port, from the vendored official pipeline (no accelerator).

The official pipeline is run as-is (vendor/hidream_models, flash-attn off as its README instructs) except:
  * steps are reduced so CPU runs finish (numbers are parity references, not quality settings);
  * cases marked "unsnapped" bypass find_closest_resolution, which otherwise forces ~4 MP sizes.
For every case the model's first calls (cond + uncond at step 0) are captured with all inputs, plus the final
patch latents z, the image and per-call timings."""
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
# FLASH_SHIM=1: run the official default two-pass flash path (causal on text, full on all) with a torch SDPA
# stand-in for flash_attn_func. No [S, S] mask is materialized, so the 8k-token edit case fits in CPU RAM.
if os.environ.get("FLASH_SHIM") == "1":
    os.environ["HIDREAM_FLASH_ATTN"] = "1"
from hidream_models import pipeline as hp  # noqa: E402
from hidream_models import qwen3_vl_transformers as qvl  # noqa: E402
from hidream_models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration  # noqa: E402


def sdpa_flash_attn_func(q, k, v, softmax_scale=None, causal=False):
    """flash_attn_func contract: q [B, S, H, D], k/v [B, S, KVH, D] (GQA) -> [B, S, H, D]."""
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal, scale=softmax_scale,
        enable_gqa=True)
    return out.transpose(1, 2)


if os.environ.get("FLASH_SHIM") == "1":
    qvl._flash_attn_func = sdpa_flash_attn_func
from transformers import AutoProcessor  # noqa: E402

SNAPSHOT = os.environ.get("HIDREAM_SNAPSHOT",
                          "/hf/hub/models--HiDream-ai--HiDream-O1-Image/snapshots/0b0901d99f200389e138c61946af1185f5f49a13")
OUT = ROOT / "golden"
P_T2I = "A red fox sitting in fresh snow at golden hour, wildlife photograph, sharp focus"
P_EDIT = "Replace the cup of coffee with a tall glass of orange juice. Keep everything else the same."
CASES = [
    {"name": "t2i2048", "prompt": P_T2I, "width": 2048, "height": 2048, "steps": 2, "seed": 32, "snap": True},
    {"name": "t2i832x640", "prompt": "A cozy reading nook with a cat sleeping on a knitted blanket, warm afternoon light",
     "width": 832, "height": 640, "steps": 20, "seed": 7, "snap": False},
    {"name": "t2i1024", "prompt": P_T2I, "width": 1024, "height": 1024, "steps": 4, "seed": 32, "snap": False},
    {"name": "edit_keep_aspect", "prompt": P_EDIT, "ref": "../flux2-klein/fixtures/croissant512.png", "steps": 2,
     "seed": 3, "keep_original_aspect": True},
]


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "12")))
    processor = AutoProcessor.from_pretrained(SNAPSHOT)
    tok = processor.tokenizer
    for name in ("boi", "bor", "eor", "bot", "tms"):
        setattr(tok, f"{name}_token", f"<|{name}_token|>")
    t0 = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(SNAPSHOT, torch_dtype=torch.bfloat16).eval()
    load_s = time.perf_counter() - t0
    selected = set(filter(None, os.environ.get("CASES", "").split(",")))
    original_snap = hp.find_closest_resolution
    for case in CASES:
        if selected and case["name"] not in selected:
            continue
        out = OUT / case["name"]
        out.mkdir(parents=True, exist_ok=True)
        calls, times = [], []
        real_forward = model.forward

        def forward(*args, **kwargs):
            started = time.perf_counter()
            result = real_forward(*args, **kwargs)
            times.append(time.perf_counter() - started)
            if len(calls) < 2:  # step 0: cond, then uncond (CFG)
                keep = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()
                        if k not in ("precomputed_image_embeds", "precomputed_deepstack_image_embeds")}
                keep["x_pred"] = result.x_pred.detach().float().clone()
                calls.append(keep)
            return result

        model.forward = forward
        hp.find_closest_resolution = original_snap if case.get("snap", True) else (lambda w, h: (w, h))
        refs = [str(ROOT / case["ref"])] if case.get("ref") else []
        started = time.perf_counter()
        image = hp.generate_image(model=model, processor=processor, prompt=case["prompt"], ref_image_paths=refs,
                                  height=case.get("height", 2048), width=case.get("width", 2048),
                                  num_inference_steps=case["steps"], guidance_scale=5.0, shift=3.0,
                                  scheduler_name="default", seed=case["seed"],
                                  keep_original_aspect=case.get("keep_original_aspect", False))
        total_s = time.perf_counter() - started
        model.forward = real_forward
        image.save(out / "image.png")
        torch.save({"calls": calls}, out / "tensors.pt")
        record = {"case": case, "attention_path": "flash-two-pass (torch SDPA shim)" if os.environ.get("FLASH_SHIM") == "1" else "4d-mask", "size": list(image.size), "load_s": load_s, "total_s": total_s, "forward_s": times,
                  "seq_len": int(calls[0]["position_ids"].shape[-1]), "txt_len": int(calls[0]["input_ids"].shape[-1]),
                  "vinput_tokens": int(calls[0]["vinputs"].shape[1]), "cfg_calls_per_step": len(calls)}
        (out / "record.json").write_text(json.dumps(record, indent=2))
        print(json.dumps({k: v for k, v in record.items() if k != "forward_s"}), flush=True)


if __name__ == "__main__":
    main()
