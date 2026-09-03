# Rule-Based

Rule-Based is a transparent multi-touch classifier built from explicit positive
and negative local-feature evidence. Each observed feature can support or
penalize each shape, and the final class scores are converted to probabilities
with softmax.

This method is useful when you want interpretable behavior between the
single-touch baseline and the learned histogram/count models.

## Runtime Behavior

The classifier accumulates hard local-feature labels. For each shape, it scores:

```text
log class prior
+ positive_weight * positive evidence from observed features
- negative_weight * negative evidence from observed features
+ repetition_weight * positive evidence from repeated observations
```

Only feature presence contributes to the main positive and negative rule terms,
so repeating the same label cannot dominate the classifier unless
`repetition_weight` is set above zero.

## Configuration

Default configuration:

```text
algorithms/rule_based/config.json
```

Model-specific sidecar:

```text
<model>.rules.json
```

Config fields:

| Field | Purpose |
| --- | --- |
| `features` | Local-feature labels expected from the encoder. |
| `shapes` | Shape classes returned by the classifier. |
| `aliases` | Accepted metadata/model label aliases mapped to canonical features. |
| `rules.<shape>.positive` | Nonnegative feature weights that support a shape. |
| `rules.<shape>.negative` | Nonnegative feature weights that penalize a shape. |
| `class_prior` | Shape prior probabilities, normalized on load. |
| `parameters.positive_weight` | Global multiplier for all positive rules. |
| `parameters.negative_weight` | Global multiplier for all negative rules. |
| `parameters.repetition_weight` | Extra weight for repeated supporting features. |
| `parameters.score_temperature` | Calibration temperature for the final softmax. |

Configuration tips:

- Give highly diagnostic features larger positive weights, such as
  `double_curvature` for `sphere` or `multi_face_vertex` for `cube`.
- Put physically incompatible features in `negative`, such as strong edge or
  apex evidence against `sphere`.
- Keep `repetition_weight` low when repeated frames from one contact may be
  similar.
- Increase `score_temperature` when the rule system is too decisive after one
  or two touches.

## Optimization

Run the unified optimizer:

```powershell
python optimize.py
```

Select **Rule-Based**, choose the trained `.pth` model and train, validation,
and optional test trial files, then run optimization.

Command-line example:

```powershell
python optimize.py --run --model models\pth\features.pth --algorithms rule_based
```

The optimizer:

1. Converts training trials into per-shape feature-presence counts.
2. Estimates signed log-odds rules for each feature and shape.
3. Splits positive and negative log-odds into the JSON rule tables.
4. Selects smoothing, `repetition_weight`, and `score_temperature` on
   validation prefix NLL.
5. Writes `optimized_config.json`, metrics, charts, and optionally installs
   `<model>.rules.json`.

Standalone module:

```powershell
python -m algorithms.rule_based.optimization --model models\pth\features.pth
```

