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

Generate hard-label optimization sequences from the annotated metadata dataset
with:

```powershell
python rfs_dataset.py
```

The tool uses the same metadata scanner and random sequence generator as the
Bayesian optimizer. It reads `custom_fields.local_feature`, lets each class's
feature vocabulary be edited, and performs a reproducible shape-stratified
train/validation/test split. Exact duplicate sequences for one shape remain in
one split. The output records contain no image metadata.

```json
[
  {
    "shape": "cone",
    "sequence": ["single_curvature", "sharp_apex", "planar"]
  }
]
```

The generated split files are:

```text
trials/rfs_train.json
trials/rfs_validation.json
trials/rfs_test.json
```

Synthetic combinations are suitable for software checks, not performance
claims. Evaluation requires independently collected object trials under a
written handheld touch policy.

## Optimize From Trials

The optimizer uses deterministic local-feature labels only. It never reads
ResNet softmax confidence or an encoder calibration sidecar. Run the complete
staged optimization with:

```powershell
python optimize.py
```

The stages are deliberately separated:

1. Training trials fit `theta_templates` with Dirichlet smoothing and
   `presence_templates` with a Beta-Bernoulli trial-presence model.
2. Validation prefixes select smoothing, relative evidence/coverage weights,
   and the unexpected-feature penalty using class-balanced negative log loss.
3. The object-level softmax temperature is fitted after score selection. This
   temperature calibrates shape belief, not the local ResNet output.
4. A validation grid selects the stopping rule with the fewest mean touches
   subject to accepted-accuracy and acceptance-rate requirements.

The original order plus reproducibly shuffled copies of each validation trial
are evaluated. Final fixed-set classification remains permutation-invariant;
the permutations measure how acquisition order affects early stopping.

Each run is saved under:

```text
optimize/<model-name>/run_<number>/rfs/
```

The RFS folder in the unified run contains the optimized config and complete
JSON/CSV evaluation report. Immutable training, validation, and optional test
snapshots are stored once in the run's `inputs/` folder. When sidecar
installation is enabled, the optimized configuration is also installed as
`features.rfs.json` beside the selected weights. `predict.py` automatically
loads this sidecar.

An untouched test file may be evaluated after parameters are frozen:

```powershell
python optimize.py
```

Do not repeatedly run test evaluation while changing parameters. That turns the
test set into another validation set.

By default, `lambda_evidence` is fixed at `1.0`. With a uniform class prior,
its absolute scale overlaps with `score_temperature`; fixing one scale makes
the remaining parameters easier to identify.

## Direct Use

```python
from algorithms.rfs import create_algorithm

accumulator = create_algorithm("models/pth/features.rfs.json")
accumulator.add_touch("planar")
result = accumulator.add_touch("straight_edge")
print(result.prediction, result.belief, result.feature_coverage)
```
