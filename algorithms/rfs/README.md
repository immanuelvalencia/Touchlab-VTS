# Hard-Label Set Evidence

This folder implements an RFS-inspired, permutation-invariant classifier for
unordered handheld tactile contacts. The local encoder is assumed to return an
accurate feature label. Its softmax confidence is neither stored nor used.

The algorithm still reports a probability over object shapes. This is not
local-model uncertainty: even a certain label such as `planar` is compatible
with a cube, cylinder, cone, or square pyramid. Shape uncertainty is resolved
by accumulating several labels.

## Input And Output

For touch number \(t\), the encoder supplies one label:

$$
f_t \in \mathcal{F}
$$

The observed finite collection after \(T\) contacts is:

$$
X_T = \{f_1, f_2, \ldots, f_T\}
$$

The score is permutation-invariant. Reordering a fixed collection produces the
same final belief, although online stopping time can change with acquisition
order.

The result contains the predicted shape, normalized shape beliefs, class
scores, binary feature coverage, shape-belief entropy, stopping decision, and
number of touches.

## Mathematics

### Frequency compatibility

For shape \(c\), the fitted categorical template is:

$$
\theta_c(f) = P(F=f \mid C=c)
$$

The average log compatibility of the observed labels with the shape is:

$$
E_T(c) = \frac{1}{T}\sum_{t=1}^{T}
\log\left(\theta_c(f_t)+\varepsilon\right)
$$

Dirichlet smoothing keeps every template entry positive. Division by \(T\)
prevents score magnitude from growing merely because more contacts were made.

### Diversity-aware coverage

A feature is covered when it has appeared at least once:

$$
m_T(f) = \mathbb{1}\left[f \in X_T\right]
$$

This is diversity-aware because twenty repeated `planar` labels cover only one
feature, while `planar`, `straight_edge`, and `multi_face_vertex` cover three
different kinds of local geometry.

The fitted presence template is:

$$
a_c(f) = P(f\text{ appears at least once in a trial}\mid C=c)
$$

Positive and unexpected observed evidence are:

$$
S_T^{+}(c)=\sum_{f\in\mathcal{F}}a_c(f)m_T(f)
$$

$$
S_T^{-}(c)=\sum_{f\in\mathcal{F}}\left(1-a_c(f)\right)m_T(f)
$$

and the coverage score is:

$$
S_T^{\mathrm{cov}}(c)=S_T^{+}(c)-\gamma S_T^{-}(c)
$$

Only observed features are scored. An expected feature that has not yet been
touched is not treated as absent.

### Shape score and belief

The combined class score is:

$$
s_T(c)=\log\left(P(C=c)+\varepsilon\right)
+\lambda_E E_T(c)+\lambda_C S_T^{\mathrm{cov}}(c)
$$

It is converted to an object-level belief using:

$$
b_T(c)=\frac{\exp\left(s_T(c)/\tau\right)}
{\sum_{c'\in\mathcal{C}}\exp\left(s_T(c')/\tau\right)}
$$

Here \(\tau\), named `score_temperature` in JSON, calibrates the final shape
scores. It is unrelated to local encoder confidence.

Shape uncertainty is measured by:

$$
H_T=-\sum_{c\in\mathcal{C}}b_T(c)\log b_T(c)
$$

### Early stopping

A result is accepted when all of these conditions hold:

$$
T\geq T_{\min},\qquad
\max_c b_T(c)\geq\eta,\qquad
H_T\leq h,\qquad
\text{prediction is stable for }W\text{ updates}
$$

At `max_touches`, a result that has not met those conditions stops as
`uncertain` rather than being presented as a confident decision.

## Configuration

`config.json` is schema version 2. A model-specific sidecar can override it:

```text
features.pth
features.txt
features.rfs.json
```

The major sections are:

- `features`: encoder label vocabulary.
- `shapes`: object classes.
- `aliases`: alternate labels mapped to canonical labels.
- `theta_templates`: per-shape feature-frequency distributions.
- `presence_templates`: per-shape trial-level feature-presence probabilities.
- `class_prior`: object-class prior.
- `parameters`: evidence, coverage, object-belief, and stopping parameters.

Version 1 probability-based files are intentionally rejected. Refit them from
hard-label trial files to avoid silently mixing two mathematical models.

## Trial Files

Prepare independently collected trials as hard-label JSON sequences. The RFS
trial recorder is not part of `predict.py`; trial collection and annotation
must be performed separately before fitting the templates.

```json
[
  {
    "shape": "cone",
    "sequence": ["single_curvature", "sharp_apex", "planar"]
  }
]
```

Store independent physical exploration trials in:

```text
trials/rfs_train.json
trials/rfs_validation.json
trials/rfs_test.json
```

Synthetic combinations are suitable for software checks, not performance
claims. Evaluation requires independently collected object trials under a
written handheld touch policy.

## Fit Templates

Fit frequency and presence templates from training trials:

```powershell
python fit_rfs_config.py `
  --trials trials/rfs_train.json `
  --model models/pth/features.pth
```

This creates the model sidecar `features.rfs.json`. No encoder calibration file
is required by the set algorithm.

## Direct Use

```python
from algorithms.rfs import create_algorithm

accumulator = create_algorithm("models/pth/features.rfs.json")
accumulator.add_touch("planar")
result = accumulator.add_touch("straight_edge")
print(result.prediction, result.belief, result.feature_coverage)
```
