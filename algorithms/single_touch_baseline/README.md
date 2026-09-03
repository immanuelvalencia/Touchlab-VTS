# Single Touch Baseline

Single Touch Baseline is the simplest shape-level reference method in the
system. It uses only the latest local-feature label produced by the encoder and
returns a shape distribution for that one touch. Previous touches are counted
for reporting, but they do not affect the next prediction.

This makes the method useful as a lower-bound comparison for multi-touch
algorithms. If a multi-touch method does not beat this baseline, the added
history is probably not helping.

## Runtime Behavior

For a predicted feature \(f_t\), the classifier reads the configured row
\(P(C \mid f_t)\), applies score temperature, and returns normalized shape
probabilities:

$$
s(c)=\log P(C=c \mid f_t)
$$

$$
P_T(C=c)=\mathrm{softmax}(s(c) / \tau)
$$

The method is not permutation invariant because there is no set to compare. It
is latest-touch invariant: two trials with the same final touch return the same
belief, regardless of earlier touches.

## Configuration

Default configuration:

```text
algorithms/single_touch_baseline/config.json
```

Model-specific sidecar:

```text
<model>.single_touch.json
```

Config fields:

| Field | Purpose |
| --- | --- |
| `features` | Local-feature labels expected from the encoder. |
| `shapes` | Shape classes returned by the classifier. |
| `aliases` | Accepted metadata/model label aliases mapped to canonical features. |
| `feature_posteriors` | One shape-probability row for each feature. Rows are normalized on load. |
| `parameters.score_temperature` | Calibration temperature. Lower values sharpen predictions; higher values soften them. |

Tune `feature_posteriors` when a feature is visually associated with different
objects for a specific sensor or encoder. Increase `score_temperature` if the
baseline is overconfident; decrease it if the correct class is usually ranked
first but with too little separation.

## Optimization

Run the unified optimizer:

```powershell
python optimize.py
```

Select **Single Touch Baseline**, choose the trained `.pth` model and train,
validation, and optional test trial files, then run optimization.

Command-line example:

```powershell
python optimize.py --run --model models\pth\features.pth --algorithms single_touch_baseline
```

The optimizer:

1. Counts every feature occurrence in the training trials.
2. Estimates \(P(C \mid f)\) with additive smoothing.
3. Selects smoothing and `score_temperature` on validation prefix NLL.
4. Writes `optimized_config.json`, metrics, charts, and optionally installs
   `<model>.single_touch.json`.

Standalone module:

```powershell
python -m algorithms.single_touch_baseline.optimization --model models\pth\features.pth
```

