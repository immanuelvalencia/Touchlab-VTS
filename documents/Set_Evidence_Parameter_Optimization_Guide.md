# Set-Based Evidence Algorithm Parameter Optimization Guide

The parameters should be optimized in stages. Optimizing everything simultaneously would increase overfitting risk and make it difficult to determine which part of the algorithm produced an improvement.

## 1. Parameters Estimated From Training Data

These quantities should be estimated directly from training data rather than selected through unrestricted hyperparameter search.

### Shape-Feature Distribution

For each object class, estimate its expected distribution of local tactile features:

$$
\theta_c(f)
=
\frac{
\alpha
+
\sum_{j:y_j=c}\sum_{t=1}^{T_j}
w_{j,t}q_{j,t}(f)
}{
|\mathcal{F}|\alpha
+
\sum_{j:y_j=c}\sum_{t=1}^{T_j}w_{j,t}
}
$$

Where:

- `c` is an object class.
- `f` is a local tactile feature.
- `y_j` is the object label of training trial `j`.
- `T_j` is the number of contacts in trial `j`.
- `q_{j,t}(f)` is the local model probability for feature `f`.
- `w_{j,t}` is the reliability weight of the contact.
- `alpha` is a positive Dirichlet smoothing value.

Only the smoothing value needs to be selected. Suggested candidates are:

```text
0.01, 0.1, 0.5, 1.0, 2.0
```

Smoothing prevents an unobserved feature from receiving an exact probability of zero.

### Local-Feature Prior

The local-feature prior is:

$$
\phi(f)=P(F=f)
$$

Estimate it from the local model's training or calibration distribution. It is also used as the baseline for feature coverage. Use a uniform prior only when the local-feature classes are genuinely balanced.

For seven balanced local-feature classes:

$$
\phi(f)=\frac{1}{7}
$$

This parameter should be estimated, not manually optimized.

### Object-Class Prior

The object-class prior is:

$$
P(C=c)
$$

Use a uniform class prior for a balanced controlled experiment:

$$
P(C=c)=\frac{1}{|\mathcal{C}|}
$$

Use nonuniform priors only when deployment frequencies are known and meaningful.

## 2. Score-Composition Parameters

The object-class score is:

$$
S(c;X_T)
=
\log\left(P(C=c)+\epsilon\right)
+
\lambda_E\bar{S}_{\mathrm{feat}}(c;X_T)
+
\lambda_C S_{\mathrm{cov}}(c;X_T)
$$

The following parameters control how this score is constructed:

| Parameter | Purpose | Suggested search values |
|---|---|---|
| `compatibility` | Selects the semantic compatibility equation | `mixture`, `expected_log` |
| `lambda_evidence` | Weight of average semantic compatibility | Fix at `1.0` |
| `lambda_coverage` | Importance of complementary feature coverage | `0, 0.1, 0.25, 0.5, 1, 2` |
| `unexpected_penalty` | Penalty for observed features incompatible with a shape | `0, 0.1, 0.25, 0.5, 1` |

### Compatibility Function

The primary mixture compatibility is:

$$
E_t^{\mathrm{mix}}(c)
=
\log\left(
\sum_{f\in\mathcal{F}}
q_t(f)\theta_c(f)
+
\epsilon
\right)
$$

The expected-log alternative is:

$$
E_t^{\log}(c)
=
\sum_{f\in\mathcal{F}}
q_t(f)
\log\left(\theta_c(f)+\epsilon\right)
$$

Use mixture compatibility as the primary implementation. A uniform local prediction contributes equal semantic compatibility to every normalized shape template. Expected-log compatibility is stricter but can favor broader templates under highly uncertain local predictions, so it should be treated as an ablation.

### Relative Score Weights

Keep:

$$
\lambda_E=1
$$

Then optimize the relative coverage contribution through `lambda_coverage`. Optimizing every score weight together with the calibration temperature creates an identifiability problem: scaling all class-score differences and changing the temperature can produce similar beliefs.

## 3. Shape-Presence Templates

The shape-presence template is:

$$
a_c(f)\in[0,1]
$$

It represents how strongly observing feature `f` supports class `c`. Unlike the shape-feature distribution, the presence values do not need to sum to one.

Do not independently optimize every shape-feature entry through unrestricted search. Five shapes and seven features would already create 35 adjustable values and a substantial overfitting risk.

Recommended approaches are:

1. Use geometry-defined values such as `0`, `0.5`, and `1`.
2. Estimate each value from the fraction of training trials in which the feature was confidently detected.
3. Compare geometry-defined and learned templates as an ablation.

Geometry-defined templates are the safest initial implementation because they preserve physical interpretability.

## 4. Local CNN Calibration

The probabilities produced by the local tactile CNN should be calibrated before they are treated as uncertainty estimates.

Given the local logits, use temperature scaling:

$$
q_t(f)
=
\frac{
\exp\left(z_t(f)/\tau_F\right)
}{
\sum_{f'\in\mathcal{F}}
\exp\left(z_t(f')/\tau_F\right)
}
$$

Optimize the local-feature temperature on a held-out local-feature calibration set by minimizing negative log-likelihood.

This step is important because low softmax entropy does not guarantee correctness. An uncalibrated neural network can be confidently wrong.

## 5. Object-Belief Calibration

After selecting the templates and score-composition parameters, calibrate the final object scores:

$$
b_T(c)
=
\frac{
\exp\left(S(c;X_T)/\tau_{\mathrm{cal}}\right)
}{
\sum_{c'\in\mathcal{C}}
\exp\left(S(c';X_T)/\tau_{\mathrm{cal}}\right)
}
$$

Optimize:

```text
calibration_temperature
```

Fit the temperature on held-out object trials by minimizing negative log-likelihood. Perform this step after all score parameters have been chosen because changing the score invalidates an earlier calibration result.

Evaluate calibration using:

- Negative log-likelihood
- Brier score
- Expected Calibration Error
- Reliability diagrams
- Calibration results separated by touch count

Until calibration is verified, the displayed percentages should be described as normalized decision scores rather than empirical class probabilities.

## 6. Early-Stopping Parameters

The system accepts a prediction only when confidence, entropy, evidence, and stability conditions are satisfied.

Confidence condition:

$$
\max_{c\in\mathcal{C}}b_T(c)
\ge
\tau_{\mathrm{stop}}
$$

Entropy condition:

$$
H_C(T)
\le
\eta
$$

Effective-evidence condition:

$$
N_{\mathrm{eff}}(T)
\ge
\nu
$$

Stability condition for a three-touch window:

$$
\hat{C}_T
=
\hat{C}_{T-1}
=
\hat{C}_{T-2}
$$

Suggested search values are:

| Parameter | Suggested search values |
|---|---|
| `stop_threshold` | `0.60` to `0.95` in increments of `0.05` |
| `entropy_threshold` | `0.2` to `1.2` nats |
| `min_effective_touches` | `2, 3, 4, 5` |
| `stability_window` | `2, 3, 4` |
| `max_touches` | Usually fixed at `8` or `10` |

The maximum number of touches is primarily an experimental or operational constraint. When it is reached before all confidence conditions pass, acquisition stops but the result remains uncertain.

### Stopping Objective

Do not optimize stopping parameters using classification accuracy alone. Evaluate:

- Accuracy across all trials
- Accuracy among accepted decisions
- Average number of touches
- Uncertain-result rate
- Incorrect confident-decision rate
- Risk-coverage curve

A useful operating-point objective is to minimize the expected number of touches:

$$
\min\mathbb{E}[T]
$$

subject to an accepted-decision accuracy requirement:

$$
\operatorname{Accuracy}_{\mathrm{accepted}}
\ge
0.95
$$

This selects the fastest configuration that satisfies the required decision reliability.

## 7. Contact-Weighting Alternatives

The safest initial contact weight is:

$$
w_t=1
$$

If a calibrated validity model is available, use:

$$
w_t=r_t
$$

An entropy-based alternative is:

$$
w_t=r_tu_t
$$

where:

$$
u_t
=
1
-
\frac{H_F(t)}{\log|\mathcal{F}|}
$$

Compare these weighting schemes as an ablation. Do not assume entropy weighting will improve performance because a neural network may be confidently incorrect.

## 8. Recommended Optimization Order

1. Train the local tactile-feature CNN.
2. Calibrate the local CNN temperature.
3. Estimate the local-feature prior.
4. Estimate the shape-feature templates.
5. Define or estimate the shape-presence templates.
6. Compare mixture and expected-log compatibility.
7. Tune coverage weight and unexpected-feature penalty.
8. Calibrate the final object-level temperature.
9. Tune the early-stopping parameters.
10. Evaluate once on the untouched test set.

## 9. Data-Splitting Requirement

Split the data by complete trials, users, and physical object instances. Never split individual tactile images from the same trial between training and validation.

An appropriate separation is:

```text
training trials
    -> learn CNN and shape templates

calibration trials
    -> calibrate local and object probabilities

validation trials
    -> select score and stopping parameters

test trials
    -> final evaluation only
```

Touch-level random splitting can leak nearly identical contacts from the same object and collection session into multiple subsets. This would produce unrealistically optimistic results.

## 10. Parameters That Should Not Be Optimized

Do not optimize the following as performance parameters:

- `eps`: keep it as a small numerical constant, such as `1e-8`.
- Feature names and object labels after the experimental protocol is fixed.
- `max_touches` solely to maximize accuracy; it represents the allowed exploration budget.
- Test-set parameters or thresholds.

The most important initial hyperparameters are therefore:

```text
alpha
compatibility
lambda_coverage
unexpected_penalty
calibration_temperature
stop_threshold
entropy_threshold
min_effective_touches
stability_window
```

The shape-feature distribution, local-feature prior, and class prior should be estimated from appropriate data rather than treated as arbitrary search parameters.
