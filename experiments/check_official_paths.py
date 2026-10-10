"""CPU baseline: how much do the official model's own two attention paths (4-D mask vs the two-pass flash
algorithm, here via a torch SDPA stand-in) disagree on a golden call? Sets the bf16 parity bar for the port."""
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
from hidream_models import qwen3_vl_transformers as qvl  # noqa: E402
from hidream_models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration  # noqa: E402

SNAPSHOT = "/hf/hub/models--HiDream-ai--HiDream-O1-Image/snapshots/0b0901d99f200389e138c61946af1185f5f49a13"


def sdpa_flash(q, k, v, softmax_scale=None, causal=False):
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal, scale=softmax_scale,
        enable_gqa=True).transpose(1, 2)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "12")))
    qvl._flash_attn_func = sdpa_flash
    model = Qwen3VLForConditionalGeneration.from_pretrained(SNAPSHOT, torch_dtype=torch.bfloat16).eval()
    out = {}
    for case in os.environ.get("CASES", "edit_keep_aspect").split(","):
        calls = torch.load(ROOT / "golden" / case / "tensors.pt", weights_only=False)["calls"]
        for i, call in enumerate(calls):
            kw = {k: v for k, v in call.items() if k != "x_pred"}
            kw["use_flash_attn"] = True
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
                x = model(**kw).x_pred.float()
            T = call["input_ids"].shape[-1]
            n = call["vinputs"].shape[1]
            sl = slice(T, T + n)
            out[f"{case}[{i}]"] = {"pcc_flash_vs_mask": pcc(x[0, sl], call["x_pred"][0, sl]),
                                   "max_abs": float((x[0, sl] - call["x_pred"][0, sl]).abs().max())}
            print(json.dumps(out), flush=True)
    (ROOT / "golden" / "official_paths.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
