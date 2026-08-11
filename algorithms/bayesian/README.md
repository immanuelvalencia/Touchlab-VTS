# Bayesian Optimization Files

The optimizer uses two stable working files:

- `touch_sequences.json`: Accumulated training sequences.
- `config.json`: Active likelihood configuration loaded by `algorithm.py`.

Each optimization also creates an immutable dated run under `runs/`:

```text
runs/
  2026-08-11_15-30-45/
    touch_sequences_2026-08-11_15-30-45.json
    bayesian_config_2026-08-11_15-30-45.json
```

The timestamp format is `YYYY-MM-DD_HH-MM-SS`, using the local system time and
an explicit 24-hour clock. The dataset snapshot records exactly what was used
to produce the matching optimized configuration.
After the dated files are written, the optimized configuration is copied to
`config.json` so the prediction application immediately uses the latest result.

JSON writes are atomic: data is first written to a temporary file and then
replaced, preventing a partial configuration if saving is interrupted.

## Metadata Dataset Generator

The optimizer's **Dataset Generator** tab scans metadata JSON files recursively.
For each object class, it collects the unique value of
`custom_fields.local_feature`. The following aliases are normalized:

```text
pyramid          -> square_pyramid
multiface_vertex -> multi_face_vertex
```

Random sequences sample with replacement from each class-specific feature
vocabulary. Repetition is allowed because separate touches can observe the same
local feature. Sequence length, sequences per class, and random seed are set in
the UI. Using the same source vocabulary, settings, and seed produces the same
sequences.

After scanning, select a class and choose **Edit Selected Features**, or
double-click its row. The editor can enable or disable any valid encoder feature
for that class. These edits affect generated sequences only; metadata files are
never modified. **Reset from Metadata** discards all edits and restores the
scanned vocabulary.

Generated files are timestamped:

```text
generated_datasets/
  touch_sequences_random_YYYY-MM-DD_HH-MM-SS.json
```

The generated file becomes the selected optimization dataset without
overwriting the manually recorded `touch_sequences.json`.

Each generated entry contains only the fields consumed by the optimizer:

```json
{
  "shape": "cube",
  "sequence": ["planar", "straight_edge", "planar"]
}
```
