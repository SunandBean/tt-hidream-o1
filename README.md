# tt-hidream-o1

**What this port adds** — the single-chip port itself, and the two upstream patches that let HiDream's own model code import and run without CUDA or flash-attn.
**What it builds on** — the [`changh95/qwen-image-2.1-p150`](https://huggingface.co/changh95/qwen-image-2.1-p150) single-chip port (Apache-2.0) and HiDream's own model code (MIT), vendored under `code/tt_hidream_o1/upstream/`.

**HiDream-O1-Image** (the full, undistilled model) ported to a single **Tenstorrent Blackhole p100a**.

Model card and demos: **[sunandbean/hidream-o1-p100a](https://huggingface.co/sunandbean/hidream-o1-p100a)**

| `…a wooden sign that says "OPEN DAILY"` | `Three red apples and two green pears` |
|:---:|:---:|
| ![](media/quality-t2i_sign.png) | ![](media/quality-t2i_count.png) |

Both 1024×1024, seed 1234, 50 steps at CFG 5, **19.4 s** on the card.

## What makes it unusual

**No VAE, no latent space.** HiDream-O1 works directly on 32×32 pixel patches. The model is the whole
of Qwen3-VL-8B with three generation parts added — `x_embedder` (pixel patch → 4096), `t_embedder1`
(timestep into the `<|tms|>` token slot) and `final_layer2` (back to a pixel patch).

It is undistilled: 50 FlowUniPC steps at CFG 5 means **100 forwards per image**. The language model
runs on the card; the vision tower, sampler and patch maths stay on the host.

## Install

```bash
pip install -e .    # inside a tt-metal / ttnn environment; ttnn is not on PyPI
```

```python
from tt_hidream_o1 import HiDreamTT, open_device, close_device

dev = open_device()
try:
    model = HiDreamTT(dev)
    image, timing = model.generate("Three red apples and two green pears on a wooden table",
                                   1024, 1024, seed=1234, snap=False)
    image.save("output.png")
finally:
    close_device(dev)
```

**`snap=False` matters.** Upstream's default rounds any request to its 4–5 MP resolution list, so
1024² would be generated at 2048².

Or over HTTP: `uvicorn tt_hidream_o1.server:app --host 0.0.0.0 --port 20000`.

## Accuracy

Against the official pipeline on CPU: first-step PCC 0.998–0.9999 and final-image PCC 0.967–0.992
across text-to-image at 832×640, 1024² and 2048², and a 2048² edit. Token ids and mRoPE positions
match exactly.

## Editing is not offered

`refs` works only at the official ~4 MP sizes. At 512–640 px every edit is noise — and so is the
**official CPU pipeline** at that size, so this is upstream behaviour, not a porting bug
(`experiments/diag_edit.py`). At 2048² the edit is correct but costs about 6 minutes.

## Layout

| Path | |
|---|---|
| `tt_hidream_o1/` | the port |
| `tt_hidream_o1/upstream/` | official model code at `2c2d29f` (MIT), flash-attn made optional, one CUDA call guarded |
| `examples/quickstart.py` | runnable example |
| `tt-model.yaml` | the container manifest the published image is built from — `tt-model package --container tt-model.yaml` |
| `PYTHON.md` | API reference |
| `experiments/` | the verification scripts behind the published numbers — most import this port under the name it had in the private tree it was written in, so read [`experiments/README.md`](experiments/README.md) before running them |

## Licence

Port code Apache-2.0; the vendored `upstream/` and the weights
([HiDream-ai/HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image)) are MIT. The
weights are not redistributed here.
