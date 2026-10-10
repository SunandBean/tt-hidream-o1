# Verification scripts

| Script | What it checks |
|---|---|
| `make_golden.py` | Records the official CPU reference the device results are compared against |
| `check_official_paths.py` | That this port takes the same code paths as the official pipeline (resolution snapping, mask construction, rope indexing) |
| `check_ref_algo.py` | The host maths — chat template, mRoPE positions, patching, the FlowUniPC schedule |
| `check_layers.py` | Per-layer output against the official modules |
| `hidream_device_check.py` | Each stage on the card and end to end; writes `device_check_result.json` |
| `quality_compare.py` | The 10-prompt and 6-edit comparison against the other models on the card |
| `cpu_edit_probe.py` | The official CPU pipeline at 512 px — the control that showed the noise is upstream |
| `diag_edit.py` | Why editing fails below ~4 MP, and that it is correct at 2048² |



## Running these outside the tree they were written in

These are the scripts as they were run, inside the private working tree this port was developed in.
They are published as the record behind the numbers on the model card, and most of them need two
edits before they will run from a clone of this repo:

1. **The package name.** Five of them (`check_layers.py`, `check_ref_algo.py`, `diag_edit.py`,
   `hidream_device_check.py`, `quality_compare.py`) do `from hidreamo1 import ...`. `hidreamo1` is
   this port, published here as **`tt_hidream_o1`**.
2. **The `sys.path` line.** The same five insert `ROOT.parents[1] / "deploy"`, the private tree's
   package directory. From a clone that is `ROOT.parent`.

`check_layers.py`, `check_official_paths.py`, `cpu_edit_probe.py` and `make_golden.py` also import
`hidream_models`, which is upstream's own [HiDream-O1-Image](https://github.com/HiDream-ai/HiDream-O1-Image)
checkout — put it on `PYTHONPATH`. The device scripts need a p100a.
