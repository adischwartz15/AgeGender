# Robustness Evaluation

Practical guide to `scripts/run_robustness.py`. For the full list of
corruption types/severities and the exact per-severity parameters, see
`docs/final_evaluation_protocol.md` ("Robustness conditions and
severities") and `configs/robustness.yaml`. For how this feeds the
multi-model comparison suite, see `docs/architecture_analysis.md`
(section 7) and `docs/backbone_comparison.md`.

## Running it

```bash
make robustness CHECKPOINT=checkpoints/<your_checkpoint>.pt
# or, explicitly:
python scripts/run_robustness.py --checkpoint <checkpoint>.pt \
    --output-dir <isolated_dir>/robustness --calibration-dir <isolated_dir>/calibration \
    [--max-samples N]
```

Evaluates the **full test split by default** (touched only for this, plus
final evaluation -- never for training or model selection) under the
deterministic corruption types x severities defined in
`configs/robustness.yaml` (currently 11 types x 3 severities = 33
conditions -- Gaussian blur, Gaussian noise, low resolution/resize
degradation, JPEG compression, low/high brightness, low/high contrast,
grayscale conversion, partial occlusion, partial crop). This count is
**never hand-maintained** -- every run computes and saves the actual count
programmatically from the config to `corruption_summary.json`
(`src/evaluation/robustness.py::corruption_summary`), and
`robustness_summary.md`'s first line always states the real count for that
run, so this document's number can never silently drift out of sync with
`configs/robustness.yaml`.

`--max-samples` deterministically **stratified-samples** by age bucket x
gender label down to about that many rows for faster iteration, instead
of truncating to whichever rows happen to sort first in the split CSV
(which could silently exclude an entire subgroup); the sampled IDs and
sampling metadata are saved to `sampling_metadata.json`. This flag is for
development iteration only -- the project's pre-registered final numbers
always use the full test split (`docs/final_evaluation_protocol.md`).

## What it reports

For every condition: age MAE, gender-label selective accuracy, abstention
rate, confidence, and interval width -- **both raw and, when a
calibration artifact is available, conformal-calibrated** (the fixed
offset from the clean calibration split, never refit per corruption --
see `docs/calibration.md`), compared against the clean baseline.

Output defaults to a checkpoint/experiment/seed-specific directory (the
same checkpoint's own isolated
`experiments/<experiment>/seed_<seed>/robustness/`, if the checkpoint
lives in that layout), never a single shared global `outputs/robustness/`
two checkpoints could silently overwrite. Saves `robustness_results.csv`,
`robustness_degradation.csv`, both performance-vs-severity and
degradation-vs-severity plots per metric, sample corrupted images, and a
Markdown summary. Every corruption is deterministic for a fixed seed
(`configs/robustness.yaml: robustness.seed`), so the same corrupted
images are shown to every model compared.

## Multi-model comparison

`src/evaluation/robustness.py` additionally provides `compute_degradation()`
(adds `{metric}_delta` and `{metric}_pct_change` columns relative to the
clean row) and `build_robustness_diff_table()` (a direct model-vs-model
difference table) -- both are wired into
`scripts/compare_backbones.py --robustness-csv` (see
`docs/backbone_comparison.md`).
