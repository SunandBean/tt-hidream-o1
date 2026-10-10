# tt_hidream_o1 — Python reference

HiDream-O1-Image (undistilled) on one Tenstorrent Blackhole p100a. The Qwen3-VL-8B language model
runs on the card; the vision tower, the sampler and the patch maths are host. There is no VAE —
the model works directly on 32×32 pixel patches.

## Install

```bash
pip install -e .                # inside a tt-metal / ttnn environment; ttnn is not on PyPI
pip install -e ".[server]"      # also fastapi / uvicorn / pydantic
```

## Weights

`HiDream-ai/HiDream-O1-Image`, pinned to `0b0901d99f200389e138c61946af1185f5f49a13`, downloaded into
the HF cache on first use. `HIDREAM_SNAPSHOT` points at a directory you manage instead.

## API

### `open_device(trace_region_size=0, l1_small_size=32768)`

Opens a 1×1 mesh with `DispatchCoreType.WORKER`. Pair with `close_device(dev)`.

### `HiDreamTT(dev, snapshot=None, prec=None)`

Loads the language model onto the card. The vision tower is built lazily on first use.

### `HiDreamTT.generate(prompt, width, height, seed, refs=None, steps=50, guidance_scale=5.0, snap=True, keep_original_aspect=False) -> (PIL.Image, Request, Timing)`

| Argument | Meaning |
|---|---|
| `width`, `height` | On a 32 px grid. Other sizes are generated larger and center-cropped. |
| `snap` | **Pass `snap=False` for the size you asked for.** The upstream default rounds the request to the official 4–5 MP resolution list, so 1024² becomes 2048². |
| `refs` | Reference images for the editing path. Only usable at the official ~4 MP sizes — see the caveat below. |
| `steps`, `guidance_scale` | 50 and 5.0. This is the undistilled model: 50 steps at CFG 5 is 100 forwards. |
| `keep_original_aspect` | Sizes the target from the reference, scaled to 2048² area. |

The middle value is the resolved `Request`: the size, patch grid and step count the pipeline
actually used after `snap` and the aspect rules were applied. Most callers want the first and
third and bind it to `_req`; the parity checks in `experiments/` read it to rebuild the same
request on the CPU.

`Timing.values` holds `vision_s`, `prefix_s`, `lm_s`.

### Lower-level entry points

| Method | |
|---|---|
| `branches(req, timing)` | builds the conditional and unconditional prefixes |
| `denoise(req, seed, steps, guidance_scale, timing)` | the sampler loop |

### `dram_stats(dev) -> dict`

`total_bytes`, `allocated_bytes`, `free_bytes`.

## The editing caveat

`refs` works, but only at the official ~4 MP sizes. At 512–640 px every edit comes out as noise — and
so does the **official CPU pipeline** at that size, so this is upstream behaviour, not a porting bug
(`experiments/diag_edit.py`). At 2048² the edit is correct and matches the CPU reference to PCC 0.9865,
but one costs about 6 minutes on the card.

## Modules

| Module | Role |
|---|---|
| `pipeline.py` | `HiDreamTT`, device open/close, timing |
| `host.py` | host maths: the chat template, mRoPE positions, patching, the schedule, request construction |
| `tt_lm.py` | Qwen3-VL-8B language model (36 layers) on the card |
| `ref_algo.py` | torch reference implementations used by the verification scripts |
| `upstream/` | official model code, vendored at `2c2d29f` (MIT), with flash-attn made optional and a CUDA call guarded |
| `config.py` | static shapes, the pinned revision, snapshot resolution |
| `server.py` | the HTTP app |
