# Bag-of-Features

This classifier treats each deterministic local-feature label as a tactile
codeword. After every touch it builds the normalized occurrence histogram and
compares that histogram with one prototype per shape using Jensen-Shannon
divergence. Negative distances, class priors, and `distance_temperature` are
converted to normalized shape probabilities with softmax.

The method is permutation invariant but retains feature multiplicity. A
model-specific configuration can be placed beside a model as `<model>.bof.json`.

For count \(n_T(f)\), the observed histogram is:

$$
h_T(f)=\frac{n_T(f)}{T}.
$$

The class score is:

$$
s_T(c)=\log P(C=c)-
\frac{D_{\mathrm{JS}}(h_T\Vert\mu_c)}{\tau_B}.
$$

Fit and validate the algorithm with:

```powershell
python optimize.py
```

Select Bag of Features in the application. Runs are saved under
`optimize/<model-name>/run_<number>/`; optional sidecar installation creates
`<model>.bof.json`.

Useful options:

```text
--smoothing 0.01,0.05,0.1,0.5,1.0
--permutations 5
--test-permutations 20
--seed 42
```

## Configuration

Default configuration:

```text
algorithms/bag_of_features/config.json
```

Model-specific sidecar:

```text
<model>.bof.json
```

Config fields:

| Field | Purpose |
| --- | --- |
| `features` | Local-feature labels expected from the encoder. |
| `shapes` | Shape classes returned by the classifier. |
| `aliases` | Accepted metadata/model label aliases mapped to canonical features. |
| `histogram_templates` | One normalized feature histogram prototype per shape. |
| `class_prior` | Shape prior probabilities, normalized on load. |
| `parameters.distance_temperature` | Distance calibration. Lower values make small histogram differences more decisive. |

Configure this method by editing the shape prototypes. Increase a feature's
weight for a shape when that feature should appear often across a complete
touch sequence. Keep small nonzero masses for rare but possible features so the
distance remains stable.

## Optimization

The optimizer fits `histogram_templates` from training trials with additive
smoothing, then selects smoothing and `distance_temperature` on validation
prefix NLL.

Command-line example:

```powershell
python optimize.py --run --model models\pth\features.pth --algorithms bag_of_features
```

Standalone module:

```powershell
python -m algorithms.bag_of_features.optimization --model models\pth\features.pth
```
