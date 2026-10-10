"""CPU: per-layer divergence between the official model (flash two-pass semantics via torch SDPA) and ref_algo on
one golden call. Compares the generation-token rows after every decoder layer (and the prefix rows)."""
import json
import os
from pathlib import Path
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT.parents[1] / "deploy"))
os.environ["HIDREAM_FLASH_ATTN"] = "1"
from hidream_models import qwen3_vl_transformers as qvl  # noqa: E402
from hidream_models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration  # noqa: E402
from hidreamo1 import host, ref_algo  # noqa: E402
from hidreamo1.config import snapshot_dir  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from check_official_paths import pcc, sdpa_flash  # noqa: E402

CASE = os.environ.get("CASE", "edit_keep_aspect")
KEEP = {0, 1, 2, 3, 5, 8, 12, 17, 23, 29, 35}


class Sparse(list):
    """Keeps only the KEEP layers (bf16) so 8k-token taps fit in RAM."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def append(self, t):
        if self.n in KEEP:
            super().append((self.n, t.to(torch.bfloat16)))
        self.n += 1
BRANCH = int(os.environ.get("BRANCH", "1"))


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "12")))
    qvl._flash_attn_func = sdpa_flash
    snap = snapshot_dir()
    call = torch.load(ROOT / "golden" / CASE / "tensors.pt", weights_only=False)["calls"][BRANCH]
    case = json.loads((ROOT / "golden" / CASE / "record.json").read_text())["case"]
    model = Qwen3VLForConditionalGeneration.from_pretrained(snap, torch_dtype=torch.bfloat16).eval()
    T0 = call["input_ids"].shape[-1] - 1
    official = Sparse()
    hooks = [layer.register_forward_hook(lambda m, i, o: official.append((o[0] if isinstance(o, tuple) else o)[0, T0:]))
             for layer in model.model.language_model.layers]
    kw = {k: v for k, v in call.items() if k != "x_pred"}
    kw["use_flash_attn"] = True
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
        model(**kw)
    for h in hooks:
        h.remove()
    sd_get = lambda k: model.state_dict()[k].to(torch.bfloat16)
    ref = ref_algo.RefHiDream(sd_get)
    processor = AutoProcessor.from_pretrained(snap)
    refs = [Image.open(ROOT / case["ref"])] if case.get("ref") else None
    req = host.build_request(processor, case["prompt"], case.get("width", 2048), case.get("height", 2048), refs=refs,
                             keep_original_aspect=case.get("keep_original_aspect", False), snap=case.get("snap", True))
    sample = req.samples[BRANCH]
    vision = None
    if sample.pixel_values is not None:
        with torch.no_grad():
            e, d = model.model.get_image_features(sample.pixel_values, sample.image_grid_thw)
        vision = (torch.cat(e), d)
    vin = call["vinputs"][0]
    taps = Sparse()
    with torch.no_grad():
        pre = ref.prefix(sample, vision)
        ref.step(sample, pre, vin[: sample.tgt_len], vin[sample.tgt_len:] if sample.ref_lens else None,
                 float(call["timestep"].reshape(-1)[0]), taps=taps)
    assert T0 == sample.prefix_len
    rows = []
    for (li, g), (_, t) in zip(official, taps):  # g: official generation rows [tms; target; refs]
        g, t = g.float(), t.float()
        rows.append({"layer": li, "gen_pcc": pcc(t, g), "tgt_pcc": pcc(t[1:1 + sample.tgt_len], g[1:1 + sample.tgt_len]),
                     "tms_pcc": pcc(t[0], g[0]), "gen_max_abs": float((t - g).abs().max()),
                     "official_absmax": float(g.abs().max())})
        print(json.dumps(rows[-1]), flush=True)
    (ROOT / "golden" / f"layers_{CASE}_{BRANCH}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
