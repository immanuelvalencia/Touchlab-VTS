# Methodology: Uncertainty-Aware Set-Based Evidence Accumulation for Handheld Visuo-Tactile Shape Recognition

## 1. Methodological Overview

This study investigates object-shape recognition from multiple unregistered tactile contacts acquired with a handheld GelSight sensor. The target objects are five geometric primitives: cube, sphere, cylinder, cone, and square pyramid. Each sensor contact produces a local visuo-tactile image. Because a single contact observes only a small portion of an object, local tactile appearance is not generally sufficient to determine the complete object shape.

For example, a planar contact can occur on a cube, the base of a cylinder, the base of a cone, or a face of a square pyramid. A single planar observation therefore provides useful evidence, but it is not an object identity. Global recognition requires the system to combine evidence from several contacts, such as a planar surface followed by a straight edge, a circular rim, a multi-face vertex, or a sharp apex.

The methodology separates recognition into two levels:

1. A convolutional neural network converts each tactile image into a probability distribution over local geometric features.
2. A set-based evidence accumulator combines a variable number of local predictions into a belief distribution over complete object shapes.

The first-level model answers the question, "What local geometry is visible in this contact?" The second-level model answers the question, "Which complete object is most compatible with the collection of local geometries observed so far?"

This decomposition is physically meaningful because it does not force an inherently local image to contain global information that is not present. It also permits uncertainty to be retained between the two levels instead of reducing every contact to a single hard label.

The proposed method was developed from an initial sequential Bayesian classifier. The Bayesian classifier is retained as a baseline because it provides a clear probabilistic interpretation and requires little computation. However, it has important limitations for handheld tactile exploration: it discards the local classifier's uncertainty, treats repeated contacts as repeated independent evidence, has no explicit representation of complementary feature coverage, and requires a complete feature-to-shape likelihood table. The proposed set-evidence method addresses these limitations using calibrated soft feature probabilities, reliability weighting, normalized semantic compatibility, diversity-aware maximum coverage, learned shape templates, and uncertainty-aware early stopping.

## 2. Terminology and Acronyms

| Term or acronym | Meaning |
|---|---|
| CNN | Convolutional Neural Network |
| RFS | Random Finite Set |
| FISST | Finite Set Statistics |
| PHD | Probability Hypothesis Density |
| GLMB | Generalized Labeled Multi-Bernoulli |
| NLL | Negative Log-Likelihood |
| ECE | Expected Calibration Error |
| TCD | Texture Contrast Difference |
| Softmax | A function that converts real-valued scores into normalized probabilities |
| Entropy | A measure of uncertainty in a probability distribution |
| Calibration | Agreement between predicted confidence and empirical correctness |
| Permutation invariance | The final output does not change when the same contacts are reordered |
| Variable cardinality | Different trials may contain different numbers of contacts |
| Evidence accumulation | Repeatedly combining information from new contacts to update an object belief |
| Diversity-aware | Rewarding complementary local evidence while preventing repeated evidence from falsely appearing diverse |

Although the implementation is referred to as the RFS or set-evidence algorithm in the software, it is important to use precise terminology in the dissertation. The method is **RFS-inspired** and **set-based**, but it is not a formal PHD, GLMB, or multi-object Bayes filter. RFS theory motivates the representation of observations as unordered finite collections with variable size. The implemented classifier then uses purpose-built semantic compatibility and coverage functions for tactile shape recognition.

## 3. Problem Formulation

### 3.1 Object and Feature Spaces

Let the object class belong to the finite class set:

$$
\mathcal{C}
=
\{\text{cube},\text{sphere},\text{cylinder},\text{cone},\text{square pyramid}\}.
$$

The local tactile feature vocabulary is:

$$
\mathcal{F}
=
\{\text{planar},\text{single curvature},\text{double curvature},
\text{straight edge},\text{circular rim},
\text{multi-face vertex},\text{sharp apex}\}.
$$

The number of object classes is denoted by:

$$
K = |\mathcal{C}|.
$$

The number of local feature classes is denoted by:

$$
F = |\mathcal{F}|.
$$

For the present study, the values are:

$$
K=5,
\qquad
F=7.
$$

### 3.2 A Trial of Handheld Contacts

A tactile trial consists of an object with class label `c` and a finite collection of tactile images:

$$
\mathcal{I}_T
=
\{I_1,I_2,\ldots,I_T\}.
$$

The number of contacts, `T`, may differ across trials. Contact position and object pose are not supplied to the classifier. Consequently, the method does not assume a registered spatial relationship among the contacts.

The desired classifier is a mapping from a finite collection of local observations to an object belief:

$$
g(\mathcal{I}_T)
\longrightarrow
b_T(c),
\qquad
c\in\mathcal{C}.
$$

The output must satisfy the probability axioms:

$$
b_T(c)\geq 0,
$$

$$
\sum_{c\in\mathcal{C}} b_T(c)=1.
$$

For a fixed contact collection, the final classifier should be permutation-invariant:

$$
g(\{I_1,I_2,\ldots,I_T\})
=
g(\{I_{\sigma(1)},I_{\sigma(2)},\ldots,I_{\sigma(T)}\}),
$$

where `sigma` is any permutation of the contact indices.

This condition applies to the final classification of a fixed collection. Early stopping is evaluated on successive prefixes, so the stopping time may change when informative contacts arrive earlier or later.

### 3.3 Why Direct Object Classification Is Insufficient

A direct single-touch classifier would attempt to estimate:

$$
P(C=c\mid I_t).
$$

This formulation assumes that one local tactile image contains sufficient evidence for a complete object class. That assumption is often false. A planar tactile image is compatible with several primitives, while a curved image may be compatible with both a cylinder and a cone. Training with only object labels also creates contradictory supervision: visually similar local contacts may be assigned different object labels even when the distinguishing global geometry is outside the sensor's field of view.

The proposed decomposition instead estimates local geometry first:

$$
P(F=f\mid I_t),
$$

and then infers the object from multiple local distributions:

$$
P(C=c\mid q_1,q_2,\ldots,q_T).
$$

This does not claim that local features uniquely identify the object. The method explicitly expects individual contacts to be ambiguous and resolves that ambiguity through complementary evidence across the trial.

## 4. Local Tactile Feature Encoder

### 4.1 Encoder Output

Each tactile image is processed by a ResNet-18 CNN. Let the unnormalized network logits for touch `t` be:

$$
\mathbf{z}_t
=
[z_{t,1},z_{t,2},\ldots,z_{t,F}]^{\mathsf{T}}.
$$

The raw softmax output is:

$$
q_t(f)
=
\frac{\exp(z_{t,f})}
{\sum_{j=1}^{F}\exp(z_{t,j})}.
$$

The vector of local feature probabilities is:

$$
\mathbf{q}_t
=
[q_t(1),q_t(2),\ldots,q_t(F)]^{\mathsf{T}}.
$$

Each component is non-negative and the components sum to one:

$$
q_t(f)\geq 0,
$$

$$
\sum_{f=1}^{F}q_t(f)=1.
$$

Unlike a hard feature label, this vector retains ambiguity. For example, a contact may contain 0.55 probability of a single-curvature feature, 0.35 probability of a circular rim, and 0.10 probability distributed among other features. The set classifier receives the complete distribution.

### 4.2 Encoder Training Objective

For a training image with ground-truth local feature label `y_i`, the standard multiclass cross-entropy loss is:

$$
\mathcal{L}_{\mathrm{enc}}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log q_i(y_i).
$$

Here, `N` is the number of training images. The labels used at this stage are local geometric features, not complete object identities. This prevents a planar face from being forced to represent one particular object.

### 4.3 Acquisition-Group-Aware Data Splitting

Video capture produces many adjacent frames from one physical contact. These frames are highly correlated and must not be treated as independent samples during dataset splitting. If neighboring frames are distributed across training, validation, and test sets, the test set can contain nearly identical images to those seen during training. The resulting accuracy is then an optimistic estimate of generalization.

The corrected split operates on acquisition groups. Let `G_j` denote one independent image capture or one complete video acquisition:

$$
G_j
=
\{I_{j,1},I_{j,2},\ldots,I_{j,n_j}\}.
$$

The dataset is partitioned at the group level:

$$
\mathcal{G}_{\mathrm{train}}
\cap
\mathcal{G}_{\mathrm{validation}}
=
\varnothing,
$$

$$
\mathcal{G}_{\mathrm{train}}
\cap
\mathcal{G}_{\mathrm{test}}
=
\varnothing,
$$

$$
\mathcal{G}_{\mathrm{validation}}
\cap
\mathcal{G}_{\mathrm{test}}
=
\varnothing.
$$

All frames belonging to the same video directory remain in one split. Data augmentation is performed only after a group has been assigned to a split. A split manifest records the group assignments for reproducibility.

### 4.4 Temperature Calibration of Encoder Probabilities

Softmax confidence is not automatically calibrated. A network can be confidently wrong, so low entropy alone cannot be interpreted as reliability until calibration has been assessed. The encoder therefore uses temperature scaling on an acquisition-group-independent validation set.

For a positive encoder temperature `T_enc`, calibrated probabilities are:

$$
\widetilde{q}_i(f;T_{\mathrm{enc}})
=
\frac{\exp(z_{i,f}/T_{\mathrm{enc}})}
{\sum_{j=1}^{F}\exp(z_{i,j}/T_{\mathrm{enc}})}.
$$

The temperature is selected by minimizing validation NLL:

$$
T_{\mathrm{enc}}^{*}
=
\underset{T_{\mathrm{enc}}>0}{\operatorname{argmin}}
\left[
-\sum_{i=1}^{N_{\mathrm{val}}}
\log \widetilde{q}_i(y_i;T_{\mathrm{enc}})
\right].
$$

A temperature larger than one softens an overconfident distribution. A temperature smaller than one sharpens an underconfident distribution. Temperature scaling does not change the largest logit and therefore normally does not change top-1 classification accuracy. It changes the confidence assigned to the prediction.

Calibration is evaluated using NLL, the Brier score, and ECE. For one-hot target vector `y_i`, the multiclass Brier score is:

$$
\operatorname{Brier}
=
\frac{1}{N}
\sum_{i=1}^{N}
\sum_{f=1}^{F}
\left(\widetilde{q}_i(f)-y_i(f)\right)^2.
$$

For confidence bins `B_m`, ECE is:

$$
\operatorname{ECE}
=
\sum_{m=1}^{M}
\frac{|B_m|}{N}
\left|
\operatorname{acc}(B_m)
-
\operatorname{conf}(B_m)
\right|.
$$

The calibration temperature and before-and-after metrics are stored in a model-specific JSON sidecar. At inference time, the prediction application loads this sidecar automatically. If it is absent, the system uses an identity temperature and disables entropy-derived reliability weighting.

For the remainder of this methodology, `q_t(f)` denotes the calibrated local probability when calibration is available and the ordinary softmax probability otherwise.

## 5. Baseline Sequential Bayesian Classifier

### 5.1 Bayesian Update

The baseline represents the object state as a posterior probability over shapes. Let the posterior after `t` contacts be:

$$
p_t(c)
=
P(C=c\mid \widehat{f}_1,\ldots,\widehat{f}_t).
$$

The baseline converts each soft encoder output into a hard local feature:

$$
\widehat{f}_t
=
\underset{f\in\mathcal{F}}{\operatorname{argmax}}
\;q_t(f).
$$

Let the feature likelihood table contain:

$$
L_{f,c}
=
P(F=f\mid C=c).
$$

Starting from a class prior `p_0(c)`, one sequential Bayes update is:

$$
p_t(c)
=
\frac{L_{\widehat{f}_t,c}\,p_{t-1}(c)}
{\sum_{c'\in\mathcal{C}}
L_{\widehat{f}_t,c'}\,p_{t-1}(c')}.
$$

After `T` touches, the recursion is equivalent to:

$$
p_T(c)
\propto
p_0(c)
\prod_{t=1}^{T}
L_{\widehat{f}_t,c}.
$$

The implementation begins with a uniform prior over the five shapes:

$$
p_0(c)=\frac{1}{5}=0.2.
$$

The optimized likelihood matrix is normalized over local features for each object class:

$$
\sum_{f\in\mathcal{F}} L_{f,c}\approx 1.
$$

### 5.2 Optimization of the Bayesian Likelihood Matrix

The Bayesian likelihood matrix can be learned from labeled object-level feature sequences. Let the unconstrained optimization matrix be:

$$
\mathbf{W}
\in
\mathbb{R}^{F\times K}.
$$

Each shape column is converted into a valid feature distribution using a softmax over the feature dimension:

$$
L_{f,c}
=
\frac{\exp(W_{f,c})}
{\sum_{g=1}^{F}\exp(W_{g,c})}.
$$

For training sequence `i`, containing hard local labels, the class score in log space is:

$$
\ell_i(c)
=
\sum_{t=1}^{T_i}
\log L_{f_{i,t},c}.
$$

Because the optimizer assumes a uniform class prior, its logarithm is the same additive constant for every class and may be omitted without changing the softmax result. The predicted sequence probability is:

$$
\widehat{p}_i(c)
=
\frac{\exp(\ell_i(c))}
{\sum_{c'\in\mathcal{C}}\exp(\ell_i(c'))}.
$$

For ground-truth object label `y_i`, the optimized objective is mean sequence-level cross-entropy:

$$
\mathcal{L}_{\mathrm{Bayes}}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log \widehat{p}_i(y_i).
$$

The current optimizer uses Adam with learning rate 0.1 for 1,000 epochs. After optimization, probability values are clamped to a minimum of 0.01 and every shape column is renormalized:

$$
\widetilde{L}_{f,c}
=
\max(L_{f,c},0.01),
$$

$$
L_{f,c}^{\mathrm{final}}
=
\frac{\widetilde{L}_{f,c}}
{\sum_{g=1}^{F}\widetilde{L}_{g,c}}.
$$

The optimizer reports mean cross-entropy rather than summed loss, making the displayed magnitude comparable across datasets with different numbers of sequences. It also reports training accuracy for interpretability.

The sequence order does not affect this optimizer because the sequence score is a sum of log likelihoods. Thus, a sequence containing planar, apex, and single-curvature labels has the same Bayesian score under any ordering. Order still matters operationally if a separate early-stopping system evaluates prefixes, but it does not affect the complete sequence posterior.

The optimization interface can scan metadata, identify unique local features per object class, permit manual editing, and generate feature sequences of different lengths. Generated combinations are useful for debugging optimization behavior. They are not a substitute for independently acquired object-level touch sequences. Training on combinations generated from a predefined shape-feature table would cause the optimized matrix to reproduce that table and would not demonstrate empirical object recognition.

### 5.3 Strengths of the Bayesian Baseline

The Bayesian method is computationally inexpensive, interpretable, and naturally supports sequential updating. Its product form is permutation-invariant for a fixed collection of hard feature labels because multiplication is commutative. It therefore provides a useful baseline for determining whether the additional uncertainty and coverage mechanisms of the set-evidence method provide measurable value.

### 5.4 Limitation 1: Hard Decisions Discard Local Uncertainty

Consider two encoder outputs with the same top feature:

$$
\mathbf{q}^{(1)}
=
[0.95,0.01,0.01,0.01,0.01,0.005,0.005],
$$

$$
\mathbf{q}^{(2)}
=
[0.24,0.22,0.18,0.12,0.10,0.08,0.06].
$$

Both are converted to the same hard label because the first component is largest. The Bayesian update therefore treats a highly confident planar prediction and a weakly ambiguous planar prediction as identical evidence. Information in the remaining probabilities is lost.

### 5.5 Limitation 2: Repeated Correlated Contacts Can Produce Overconfidence

The product update is derived under conditional independence of observations given the class:

$$
P(\widehat{f}_1,\ldots,\widehat{f}_T\mid C=c)
=
\prod_{t=1}^{T}
P(\widehat{f}_t\mid C=c).
$$

Handheld contacts may violate this assumption. Adjacent touches can revisit the same region, and adjacent video frames can represent the same physical contact. If the same feature is repeatedly observed, its likelihood is repeatedly multiplied:

$$
p_T(c)
\propto
p_0(c)
L_{f,c}^{T}.
$$

Even modest differences between classes become extreme as `T` increases. The posterior can therefore become very confident without receiving new geometric information.

### 5.6 Limitation 3: Frequency Is Not the Same as Complementary Coverage

A sequence containing five planar observations has more measurements than a sequence containing one planar observation, but it may not contain more kinds of geometric evidence. The baseline accumulates frequency. It does not explicitly distinguish repeated evidence from complementary evidence.

This matters for objects that share local features. Cube and square pyramid both produce planar surfaces and straight edges. A distinctive multi-face vertex or sharp apex can be more informative than several additional planar contacts. The baseline has no separate mechanism for representing whether a distinctive feature has appeared at least once.

### 5.7 Limitation 4: Complete Likelihood Encoding Is Required

The Bayesian classifier requires every feature-shape likelihood:

$$
L_{f,c}=P(F=f\mid C=c).
$$

These values must be manually specified or estimated from data. They are also dependent on the contact policy. More precisely, the observed feature distribution is:

$$
P(F=f\mid C=c,\pi),
$$

where `pi` denotes how contacts are selected. A participant who mainly touches broad faces produces a different distribution from a participant who deliberately searches for edges and vertices. A likelihood table learned under one policy may not transfer to another.

The RFS-inspired method cannot eliminate this dependence because it is an information property of tactile exploration. It does, however, make the policy dependence explicit, learns its templates from object-level trials, retains soft uncertainty, and separates frequency compatibility from feature presence.

### 5.8 Limitation 5: No Explicit Uncertainty-Aware Stopping Rule

The baseline implementation continuously updates its posterior but does not require a minimum amount of reliable evidence, prediction stability, or an uncertainty threshold before accepting a result. It also has no explicit uncertain outcome when the maximum touch budget is exhausted.

## 6. Random Finite Set Motivation

### 6.1 Definition of an RFS

A Random Finite Set is a random variable whose realization is a finite set. Both the number of elements and the element values are random:

$$
X
=
\{x_1,x_2,\ldots,x_N\},
$$

where the cardinality `N` is random:

$$
N=|X|.
$$

In robotics, RFS methods are commonly used when the number of entities is unknown or changes over time, such as multi-target tracking, landmark mapping, occupancy estimation, and data association. A major attraction is that an RFS represents the collection directly without assigning an arbitrary order to its elements.

Under an independent and identically distributed cluster RFS, let the cardinality distribution be:

$$
\rho(n)=P(|X|=n),
$$

and let the single-element spatial density be `s(x)`. Under the standard FISST set-integral convention, the finite-set density is:

$$
f(X)
=
|X|!\,\rho(|X|)
\prod_{x\in X}s(x).
$$

The factorial appears because the FISST set integral contains the reciprocal factorial:

$$
\int f(X)\,\delta X
=
f(\varnothing)
+
\sum_{n=1}^{\infty}
\frac{1}{n!}
\int
f(\{x_1,\ldots,x_n\})
\,dx_1\cdots dx_n.
$$

This formalism is useful for understanding unordered variable-cardinality data. However, the present task is not multi-object tracking: every trial contains one physical object, while the elements of the set are observations of that object. A full RFS Bayes filter would therefore add machinery that is not required for the research question.

### 6.2 Adaptation to Tactile Recognition

The tactile observation collection is represented conceptually as:

$$
X_T
=
\{(\mathbf{q}_1,w_1),(\mathbf{q}_2,w_2),\ldots,(\mathbf{q}_T,w_T)\}.
$$

Each element contains a local feature-probability vector and a reliability weight. The collection has variable cardinality and its final score is permutation-invariant. Instead of propagating a formal multi-object RFS density, the method computes discriminative semantic evidence over object classes.

The RFS contribution should therefore be described as follows:

> RFS principles motivate the unordered, variable-cardinality representation of tactile observations. The implemented method uses a specialized set-based discriminative accumulator rather than a formal RFS tracking filter.

## 7. Proposed Uncertainty-Aware Set-Evidence Method

### 7.1 Meaning of Uncertainty-Aware

The method is uncertainty-aware in three distinct ways.

First, it retains the full calibrated feature distribution rather than only its maximum component. Second, it computes a bounded reliability weight from normalized feature entropy, but only when encoder calibration is available. Third, it can reject a trial as uncertain when confidence, entropy, evidence quantity, and stability requirements are not satisfied before the touch budget is exhausted.

### 7.2 Local Feature Entropy

The uncertainty of one calibrated local prediction is measured by Shannon entropy:

$$
H(\mathbf{q}_t)
=
-\sum_{f=1}^{F}
q_t(f)\log q_t(f).
$$

The maximum entropy occurs for a uniform distribution:

$$
H_{\max}=\log F.
$$

A normalized certainty score is therefore:

$$
u_t
=
1-
\frac{H(\mathbf{q}_t)}{\log F}.
$$

The range is:

$$
0\leq u_t\leq 1.
$$

A uniform prediction gives certainty near zero. A one-hot prediction gives certainty near one.

### 7.3 Conservative Reliability Weight

When a valid encoder-calibration sidecar is present, touch reliability is:

$$
w_t
=
w_{\min}
+
(1-w_{\min})u_t^{\gamma_w}.
$$

Here, `w_min` is the minimum touch weight and `gamma_w` controls how rapidly weight increases with certainty. The default values are:

$$
w_{\min}=0.25,
$$

$$
\gamma_w=1.
$$

Consequently:

$$
w_{\min}\leq w_t\leq 1.
$$

If calibration is unavailable, entropy is not assumed to measure reliability and the implementation uses:

$$
w_t=1.
$$

The soft vector still expresses ambiguity in this fallback case, but no additional entropy-based attenuation is applied. This behavior is intentionally conservative because an uncalibrated network may be confidently wrong.

The effective amount of accumulated evidence is:

$$
N_{\mathrm{eff}}(T)
=
\sum_{t=1}^{T}w_t.
$$

This quantity is used by the normalized evidence score and the minimum-evidence stopping requirement.

### 7.4 Shape-Specific Feature-Frequency Template

For each shape, the model contains a normalized semantic feature template. Because the fitted values are averages of encoder posterior probabilities, the exact implemented definition is:

$$
\theta_c(f)
=
\mathbb{E}
\left[
q_t(f)
\mid C=c,\pi
\right].
$$

The touch-policy symbol is included because feature frequencies depend on how the object is explored. Each template satisfies:

$$
\theta_c(f)>0,
$$

$$
\sum_{f=1}^{F}\theta_c(f)=1.
$$

These templates describe typical local semantic mass under the encoder and touch policy. With a well-calibrated feature encoder they can be interpreted as soft estimates of feature prevalence, but they are not direct image likelihoods. The local CNN output is a posterior over features, so the method uses a semantic compatibility function rather than claiming to compute a generative image likelihood.

### 7.5 Per-Touch Semantic Compatibility

The default implementation uses mixture compatibility:

$$
e_t^{\mathrm{mix}}(c)
=
\log
\left(
\sum_{f=1}^{F}
q_t(f)\theta_c(f)
+
\varepsilon
\right).
$$

This is the logarithm of the overlap between the local feature distribution and the class template. It is forgiving when probability mass is shared among several compatible features.

An alternative expected-log compatibility is also implemented:

$$
e_t^{\mathrm{elog}}(c)
=
\sum_{f=1}^{F}
q_t(f)
\log\left(\theta_c(f)+\varepsilon\right).
$$

The expected-log form penalizes disagreement more strongly because probability assigned to a feature with low template probability contributes a large negative value. The mixture form is the current default and should be compared with expected-log compatibility as an ablation.

Neither expression is identified as:

$$
\log P(I_t\mid C=c).
$$

They are discriminative semantic compatibility scores constructed from a feature posterior and a class template.

### 7.6 Normalized Evidence Accumulation

The weighted semantic evidence is normalized by effective touch count:

$$
\overline{E}_T(c)
=
\frac{
\sum_{t=1}^{T}w_t e_t(c)
}
{
N_{\mathrm{eff}}(T)+\varepsilon
}.
$$

This normalization prevents the magnitude of the semantic term from growing automatically with the number of contacts. It also keeps the relative meaning of the evidence and coverage coefficients more stable across different touch counts.

Repeated identical contacts therefore do not make the normalized compatibility increasingly extreme. They may increase effective evidence count for stopping, but they do not continually multiply the class odds as in the baseline Bayesian product.

### 7.7 Meaning of Diversity-Aware Coverage

In this method, diversity does not mean visual color diversity or a generic distance between embeddings. It means **coverage of different semantically meaningful local geometries**.

For example, three planar contacts mostly repeat one type of information. A set containing a planar contact, a straight edge, and a sharp apex covers three different geometric features. The latter collection is more useful for separating a square pyramid from a cube, even if both collections contain three contacts.

The method uses a maximum operator for each feature. Once strong evidence of a feature has been observed, repeated observations of that same feature do not continually increase its coverage. This gives the method its diversity-aware behavior.

### 7.8 Prior-Corrected Maximum Feature Coverage

Let `r(f)` denote the global baseline probability of feature `f`. Raw probability above this baseline is converted to feature coverage:

$$
m_T(f)
=
\max_{1\leq t\leq T}
\left[
w_t
\cdot
\max
\left(
\frac{q_t(f)-r(f)}{1-r(f)},
0
\right)
\right].
$$

The implementation clamps the result to the unit interval:

$$
m_T(f)
\leftarrow
\min\left(\max(m_T(f),0),1\right).
$$

The baseline correction ensures that a uniform or prior-level probability does not count as positive feature coverage. If the prior is uniform, then:

$$
r(f)=\frac{1}{F}.
$$

This maximum-coverage rule was selected instead of a noisy-OR expression. Under noisy-OR, many weak feature probabilities could accumulate into false certainty. For example, repeatedly assigning 0.10 probability to an apex could eventually imply that an apex was almost certainly observed. Maximum coverage avoids that inflation: repeated weak support remains weak.

### 7.9 Trial-Level Feature-Presence Template

Feature frequency and feature presence represent different information. A feature may be rare per touch but highly diagnostic when it appears at least once. The presence parameter is:

$$
a_c(f)
=
P(
\text{feature }f\text{ is observed at least once in a trial}
\mid C=c,\pi
).
$$

Unlike the initial binary templates, the fitted implementation represents `a_c(f)` as a continuous probability between zero and one. This more accurately expresses that a feature may be common, occasional, or rare under the selected touch policy.

### 7.10 Positive and Unexpected Coverage

Positive coverage compatibility is:

$$
S_{\mathrm{positive}}(c;X_T)
=
\sum_{f=1}^{F}
a_c(f)m_T(f).
$$

Unexpected observed evidence is:

$$
S_{\mathrm{unexpected}}(c;X_T)
=
\sum_{f=1}^{F}
\left(1-a_c(f)\right)m_T(f).
$$

The complete coverage score is:

$$
S_{\mathrm{cov}}(c;X_T)
=
S_{\mathrm{positive}}(c;X_T)
-
\gamma_{\mathrm{unexpected}}
S_{\mathrm{unexpected}}(c;X_T).
$$

The coefficient `gamma_unexpected` controls the penalty for an observed feature that is unusual for a class.

The score deliberately does **not** penalize an expected feature merely because it has not been observed. Under unregistered handheld exploration, absence of observation is not evidence that the feature does not exist. A cone may not produce an apex observation if the participant never touches the apex. A valid absence penalty would require an explicit detection model conditioned on the object, touch count, and exploration policy.

### 7.11 Complete Class Score

The discriminative score for object class `c` is:

$$
S_T(c)
=
\log\left(P(C=c)+\varepsilon\right)
+
\lambda_E\overline{E}_T(c)
+
\lambda_C S_{\mathrm{cov}}(c;X_T).
$$

Here, `lambda_E` controls the contribution of average semantic compatibility and `lambda_C` controls the contribution of complementary feature coverage.

The current default parameters are:

$$
\lambda_E=1.0,
$$

$$
\lambda_C=0.5,
$$

$$
\gamma_{\mathrm{unexpected}}=0.25.
$$

These defaults are initial values, not final empirical results. They must be validated through ablation and optimization on training or validation data only.

### 7.12 Object Belief

The class scores are converted to a normalized belief using a score temperature `T_cls`:

$$
b_T(c)
=
\frac{
\exp\left(S_T(c)/T_{\mathrm{cls}}\right)
}
{
\sum_{c'\in\mathcal{C}}
\exp\left(S_T(c')/T_{\mathrm{cls}}\right)
}.
$$

The predicted shape is:

$$
\widehat{c}_T
=
\underset{c\in\mathcal{C}}{\operatorname{argmax}}
\;b_T(c).
$$

The score temperature controls the sharpness of the final object belief. It is separate from the encoder temperature. The encoder temperature calibrates local feature probabilities, while the score temperature scales the global shape belief.

### 7.13 Global Belief Entropy

The uncertainty of the object belief is:

$$
H_T^{C}
=
-\sum_{c\in\mathcal{C}}
b_T(c)\log b_T(c).
$$

Its valid range is:

$$
0\leq H_T^{C}\leq \log K.
$$

A low value indicates that belief is concentrated on a small number of classes. A high value indicates unresolved ambiguity.

### 7.14 Early-Stopping Rule

Acceptance requires four conditions:

1. Sufficient effective evidence has been accumulated.
2. The largest object belief exceeds a confidence threshold.
3. Global belief entropy is below an uncertainty threshold.
4. The predicted class has remained stable for a specified number of updates.

Define the confidence condition:

$$
A_T
=
\mathbb{I}
\left[
N_{\mathrm{eff}}(T)\geq N_{\min}
\right]
\mathbb{I}
\left[
\max_{c}b_T(c)\geq \tau_b
\right]
\mathbb{I}
\left[
H_T^{C}\leq \tau_H
\right].
$$

Define prediction stability over a window of `W` updates:

$$
R_T
=
\mathbb{I}
\left[
\widehat{c}_{T-W+1}
=
\widehat{c}_{T-W+2}
=
\cdots
=
\widehat{c}_T
\right].
$$

The system accepts when:

$$
A_T R_T=1.
$$

The complete stopping decision is:

$$
\operatorname{Stop}(T)
=
\begin{cases}
\text{accept }\widehat{c}_T,
& \text{if }A_T R_T=1,\\
\text{uncertain},
& \text{if }T\geq T_{\max}\text{ and acceptance has not occurred},\\
\text{continue},
& \text{otherwise}.
\end{cases}
$$

In the implementation, confidence acceptance has priority if its conditions become satisfied exactly at the maximum-touch boundary. Otherwise, reaching the maximum produces an uncertain result.

The initial default stopping values are:

$$
\tau_b=0.85,
$$

$$
\tau_H=0.55,
$$

$$
N_{\min}=3,
$$

$$
W=3,
$$

$$
T_{\max}=8.
$$

The stopping thresholds are selected on validation trials rather than fixed from test performance.

## 8. Learning the Set-Evidence Templates

### 8.1 Object-Level Training Trials

Template fitting uses independent object-level touch trials. Let `R_c` be the number of training trials for class `c`, and let trial `r` contain `T_{c,r}` contacts:

$$
X_{c,r}
=
\{\mathbf{q}_{c,r,1},\ldots,\mathbf{q}_{c,r,T_{c,r}}\}.
$$

The preferred input is the complete calibrated local probability vector for every contact. Hard labels are supported as one-hot vectors, but they discard uncertainty.

### 8.2 Dirichlet-Smoothed Feature Frequency

For a symmetric Dirichlet prior with concentration `alpha_theta`, the estimated class feature template is:

$$
\widehat{\theta}_c(f)
=
\frac{
\alpha_{\theta}
+
\sum_{r=1}^{R_c}
\sum_{t=1}^{T_{c,r}}
q_{c,r,t}(f)
}
{
F\alpha_{\theta}
+
\sum_{r=1}^{R_c}T_{c,r}
}.
$$

Because every probability vector sums to one, the denominator is the prior mass plus the number of touches. Dirichlet smoothing prevents zero template probabilities, which would otherwise produce undefined logarithms and excessive penalties for unseen features.

The default prior concentration is:

$$
\alpha_{\theta}=1.
$$

### 8.3 Global Feature Prior

The global feature baseline used in coverage is estimated from all training contacts:

$$
\widehat{r}(f)
=
\frac{
\alpha_{\theta}
+
\sum_{c\in\mathcal{C}}
\sum_{r=1}^{R_c}
\sum_{t=1}^{T_{c,r}}
q_{c,r,t}(f)
}
{
F\alpha_{\theta}
+
\sum_{c\in\mathcal{C}}
\sum_{r=1}^{R_c}T_{c,r}
}.
$$

This baseline prevents frequently predicted features from receiving the same novelty credit as features whose probabilities rise clearly above their normal rate.

### 8.4 Beta-Bernoulli Presence Estimation

For a presence-detection threshold `tau_p`, define whether feature `f` was confidently observed at least once in training trial `r`:

$$
d_{c,r}(f)
=
\mathbb{I}
\left[
\max_{1\leq t\leq T_{c,r}}
q_{c,r,t}(f)
\geq
\tau_p
\right].
$$

With a Beta prior having parameters `alpha_a` and `beta_a`, the posterior-mean presence estimate is:

$$
\widehat{a}_c(f)
=
\frac{
\sum_{r=1}^{R_c}d_{c,r}(f)
+
\alpha_a
}
{
R_c+\alpha_a+\beta_a
}.
$$

The defaults are:

$$
\tau_p=0.5,
$$

$$
\alpha_a=0.5,
$$

$$
\beta_a=0.5.
$$

This is a Jeffreys-style symmetric Beta prior. It avoids brittle binary templates and prevents estimates of exactly zero or one when the number of trials is small.

### 8.5 Class Prior

The implementation uses a uniform class prior:

$$
P(C=c)=\frac{1}{K}.
$$

A non-uniform prior may be used in an application with known class prevalence, but a uniform prior is appropriate for balanced experimental comparison because it prevents collection frequency from becoming a hidden source of class preference.

## 9. Complete Set-Evidence Algorithm

For each new tactile contact, the implemented algorithm performs the following steps:

1. Acquire tactile image `I_t`.
2. Compute encoder logits.
3. Apply encoder temperature calibration if available.
4. Compute the complete local feature distribution `q_t`.
5. Compute local entropy and reliability weight `w_t` when calibration is available.
6. Add the pair containing `q_t` and `w_t` to the finite observation collection.
7. Compute normalized semantic compatibility for every object class.
8. Update maximum coverage for every local feature.
9. Compute positive and unexpected coverage for every object class.
10. Combine prior, semantic compatibility, and coverage into a class score.
11. Convert class scores into a normalized object belief.
12. Evaluate confidence, entropy, effective evidence, stability, and maximum-touch conditions.
13. Accept, continue, or return uncertain.

Algorithmically, the process is:

```text
Input:
    calibrated local encoder
    shape frequency templates
    shape presence templates
    class and feature priors
    scoring and stopping parameters

Initialize:
    touch collection = empty
    prediction history = empty

For each contact t:
    q_t = calibrated feature probabilities from tactile image I_t

    if encoder calibration is available:
        w_t = entropy-derived reliability
    else:
        w_t = 1

    add (q_t, w_t) to the collection
    compute effective touch count
    compute normalized semantic compatibility for each shape
    compute prior-corrected maximum feature coverage
    compute positive and unexpected coverage for each shape
    compute class scores and softmax beliefs
    select the highest-belief shape

    if confidence, entropy, evidence, and stability criteria hold:
        return accepted shape and belief

    if maximum touch count is reached:
        return uncertain and the best current hypothesis

Return current belief and request another touch
```

## 10. How the Proposed Method Addresses the Bayesian Problems

| Bayesian limitation | Set-evidence response |
|---|---|
| Hard top-feature decision | Uses the complete calibrated feature distribution |
| Confident and ambiguous contacts treated identically | Retains soft uncertainty and optionally applies calibrated entropy weighting |
| Repeated evidence multiplied indefinitely | Uses normalized compatibility and maximum feature coverage |
| No distinction between repetition and complementarity | Tracks whether different semantic features have been strongly covered |
| No explicit unexpected-feature mechanism | Penalizes observed features that are improbable under a shape's presence template |
| Missing expected feature can be mistaken for negative evidence in naive coverage models | Does not penalize expected features merely because they were not touched |
| Manually encoded likelihood matrix | Learns frequency and presence templates from object-level trials |
| No reliable stopping policy | Uses effective evidence, confidence, entropy, stability, and a maximum touch budget |
| No uncertain outcome | Returns uncertain if acceptance conditions are not reached |
| Fixed confidence interpretation | Calibrates local encoder probabilities on independent validation groups |

The method does not remove the need for geometric structure. It replaces a single manually specified likelihood matrix with two empirically interpretable templates: local feature frequency and trial-level feature presence. It also does not make fundamentally indistinguishable shapes identifiable. If two objects generate the same distribution of observable local features under the same touch policy, no pose-free local method can separate them reliably without adding global information.

## 11. Parameter Selection and Optimization

### 11.1 Parameters Learned From Training Trials

The following quantities are estimated from object-level training trials only:

| Quantity | Interpretation |
|---|---|
| `theta_c(f)` | Expected per-touch feature frequency for shape `c` |
| `a_c(f)` | Probability that feature `f` appears at least once in a trial of shape `c` |
| `r(f)` | Global feature baseline |
| `P(C=c)` | Class prior, currently fixed to uniform |

### 11.2 Encoder Calibration Parameter

The encoder temperature is fitted on the grouped image validation split:

$$
T_{\mathrm{enc}}^{*}
=
\operatorname{argmin}
\mathcal{L}_{\mathrm{NLL,val}}.
$$

It must not be fitted on the test set.

### 11.3 Stopping Parameters

The following parameters are searched on object-level validation trials:

| Parameter | Meaning |
|---|---|
| `tau_b` | Minimum accepted object confidence |
| `tau_H` | Maximum accepted object entropy |
| `N_min` | Minimum effective touch count |
| `W` | Required prediction-stability window |

The validation tuner evaluates candidate combinations over repeated random permutations of every trial. A candidate is considered feasible when its accuracy among accepted predictions reaches a target, currently 0.95. Among feasible candidates, selection prioritizes higher acceptance rate and then fewer mean touches. Test trials are evaluated only after selection.

### 11.4 Parameters Requiring Ablation

The following scoring parameters should be selected conservatively and reported through ablation rather than optimized without constraint:

| Parameter | Meaning |
|---|---|
| `lambda_E` | Weight of normalized semantic compatibility |
| `lambda_C` | Weight of feature coverage |
| `gamma_unexpected` | Penalty for unexpected observed features |
| `w_min` | Minimum reliability assigned to an uncertain calibrated contact |
| `gamma_w` | Shape of the certainty-to-weight mapping |
| `T_cls` | Temperature of the final object belief |
| Compatibility mode | Mixture or expected-log semantic compatibility |

The primary comparison should include both unweighted and calibrated entropy-weighted variants because confidence weighting is not guaranteed to improve performance.

## 12. Experimental Methodology

### 12.1 Experimental Units

Three different experimental units must remain distinct:

1. **Frame:** one image from the sensor.
2. **Contact acquisition:** one physical contact, possibly represented by multiple adjacent video frames.
3. **Object trial:** one complete sequence or set of contacts used to classify one object.

Frames from the same contact are not independent. Contacts from the same object trial are also not independent trials. Reported sample sizes should state all three quantities to avoid overstating the amount of independent data.

### 12.2 Encoder Dataset

The encoder dataset contains tactile images labeled by local feature. Splitting is performed by acquisition group, not frame. Where possible, evaluation should also hold out physical object instances, collection sessions, or participants to test generalization beyond one sensor placement and one operator.

### 12.3 Object-Level Trial Dataset

The RFS training, validation, and test files contain complete touch trials. The same trial must never appear in more than one split. Stronger evaluation holds out collection sessions, participants, or physical replicas rather than merely shuffling trials from the same session.

The exploration policy must be explicitly stated. The term "random touch" should be used only if contact locations are randomized by a defined procedure. For participant-controlled collection, more accurate descriptions include "unregistered contacts," "uncontrolled contacts," or "natural handheld exploration."

A defensible protocol should specify:

- whether participants can see the object;
- whether they are instructed to seek diverse geometry;
- whether repeated locations are discouraged;
- the maximum number of contacts;
- whether the object or sensor is handheld;
- whether the same physical object instance appears across splits;
- whether contact order is preserved in storage;
- how invalid or non-contact frames are rejected before classification.

### 12.4 Trial Storage

Each object trial stores the true shape and the sequence of complete calibrated feature distributions. A conceptual example is:

```json
{
  "shape": "cone",
  "sequence": [
    {
      "planar": 0.03,
      "single_curvature": 0.72,
      "double_curvature": 0.02,
      "straight_edge": 0.02,
      "circular_rim": 0.16,
      "multi_face_vertex": 0.01,
      "sharp_apex": 0.04
    },
    {
      "planar": 0.02,
      "single_curvature": 0.11,
      "double_curvature": 0.01,
      "straight_edge": 0.01,
      "circular_rim": 0.04,
      "multi_face_vertex": 0.01,
      "sharp_apex": 0.80
    }
  ]
}
```

Synthetic combinations generated from known shape-feature rules are useful for unit testing and debugging. They must not be used as the final experimental dataset because they reproduce the same assumptions encoded in the templates and would create circular evaluation.

### 12.5 Baselines

The minimum baseline set should include:

1. Single-touch object prediction using the first contact.
2. Majority vote over hard local feature predictions with an object mapping.
3. Sequential Bayesian fusion using hard top-feature labels.
4. Mean pooling of local feature probability vectors followed by a classifier.
5. The proposed set-evidence method without coverage.
6. The full proposed method.

A learned Deep Sets baseline may be added if sufficient object-level trials are available. It would test whether the interpretable semantic accumulator is competitive with a fully learned permutation-invariant model.

### 12.6 Required Ablations

The following ablations isolate the contribution of each component:

- hard labels versus full feature distributions;
- no encoder calibration versus temperature calibration;
- unit weights versus calibrated entropy weights;
- summed evidence versus normalized evidence;
- compatibility only versus compatibility plus coverage;
- no unexpected-feature penalty versus the complete coverage score;
- maximum coverage versus noisy-OR coverage;
- mixture compatibility versus expected-log compatibility;
- fixed touch count versus uncertainty-aware early stopping;
- handcrafted templates versus templates fitted from real trials.

### 12.7 Permutation Evaluation

Final set classification is permutation-invariant, but stopping time depends on acquisition order because the prefixes differ. Each held-out trial is therefore randomly permuted several times. For trial `r`, let the observed stopping times over `M` permutations be:

$$
T_{r,1}^{*},T_{r,2}^{*},\ldots,T_{r,M}^{*}.
$$

The mean stopping time is:

$$
\overline{T}^{*}
=
\frac{1}{RM}
\sum_{r=1}^{R}
\sum_{m=1}^{M}
T_{r,m}^{*}.
$$

The distribution, standard deviation, and selected percentiles should also be reported because the mean alone can hide order sensitivity.

### 12.8 Performance Metrics

Overall classification accuracy is:

$$
\operatorname{Accuracy}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbb{I}[\widehat{c}_i=c_i].
$$

Acceptance rate is:

$$
\operatorname{AcceptanceRate}
=
\frac{N_{\mathrm{accepted}}}{N}.
$$

Uncertain or rejection rate is:

$$
\operatorname{UncertainRate}
=
1-\operatorname{AcceptanceRate}.
$$

Accuracy among accepted predictions is:

$$
\operatorname{AcceptedAccuracy}
=
\frac{
\sum_{i=1}^{N}
\mathbb{I}[\text{accepted}_i]
\mathbb{I}[\widehat{c}_i=c_i]
}
{
\sum_{i=1}^{N}
\mathbb{I}[\text{accepted}_i]
}.
$$

Mean touch count is:

$$
\operatorname{MeanTouches}
=
\frac{1}{N}
\sum_{i=1}^{N}T_i^{*}.
$$

Accuracy should also be reported as a function of available contacts:

$$
\operatorname{Accuracy}(t)
=
\frac{1}{N_t}
\sum_{i=1}^{N_t}
\mathbb{I}[\widehat{c}_{i,t}=c_i].
$$

Confusion matrices, macro-averaged precision, recall, and F1-score should accompany accuracy, especially if trial counts are imbalanced. Global belief calibration should be reported using reliability diagrams, ECE, NLL, and the Brier score.

### 12.9 Statistical Reporting

Performance should be reported over independent trial groups rather than treating random permutations as new independent physical trials. Confidence intervals can be computed by bootstrapping at the object-trial level. When multiple users or object instances are available, hierarchical or grouped bootstrap sampling should preserve those dependencies.

## 13. Software and Methodological Changes Implemented

### 13.1 Dynamic Algorithm Architecture

The shape algorithms were separated into dedicated folders and registered through a central algorithm registry. The prediction application constructs algorithm selectors and comparison controls from this registry. New algorithms can therefore be added without placing their mathematics directly inside the user-interface code.

The Bayesian implementation is located under `algorithms/bayesian`, while the set-evidence implementation is located under `algorithms/rfs`.

### 13.2 Bayesian Optimization and Dataset Tools

The Bayesian optimizer was extended to scan tactile metadata and collect the unique `local_feature` values associated with each object class. The discovered feature lists can be edited before sequence generation. The generator supports different sequence lengths and reproducible random generation, while optimizer inputs are validated before training.

Bayesian likelihood optimization now uses mean sequence-level cross-entropy, reports training accuracy, constrains each class column to a valid feature distribution, and writes both a dated result and an active runtime configuration. Artifact timestamps use local system time in 24-hour notation. Generated touch-sequence files do not contain unnecessary per-sequence metadata.

### 13.3 Dynamic RFS Configuration

Features, shapes, aliases, frequency templates, presence templates, priors, evidence parameters, reliability parameters, and stopping parameters are loaded from JSON. The mathematical implementation contains no hardcoded shape or feature vocabulary.

A model-specific configuration uses the same base filename as the tactile encoder:

```text
features.pth
features.txt
features.calibration.json
features.rfs.json
```

The prediction application automatically discovers these files. This allows a newly trained encoder with a different feature vocabulary or object class set to use its own configuration without editing the accumulator mathematics.

### 13.4 RFS Touch-Trial Recorder

The prediction application now contains a dedicated RFS Data tab for collecting genuine object-level trials. It reuses the selected local-feature model, encoder calibration, contact-area detector, hold timer, and contact-removal gate. The operator selects the train, validation, or test split and the true object shape before starting a trial. Every accepted physical contact stores the complete calibrated feature distribution rather than only the highest-probability label.

The interface displays touch number, top local feature, confidence, entropy, and the complete probability vector. The latest touch can be removed before saving. Shape, split, and output directory are captured when the trial starts, preventing an accidental selector change from relabeling an active trial. Model aliases are converted to canonical RFS labels, including conversion of `multiface_vertex` to `multi_face_vertex`.

Saved files contain only the object shape and probability sequence required by fitting and evaluation. Separate `rfs_train.json`, `rfs_validation.json`, and `rfs_test.json` files are updated atomically so interruption cannot leave a partially written trial file.

### 13.5 Feature Vocabulary Revision

The earlier curvature-gradient category was removed because it was difficult to distinguish consistently from single curvature in tactile images. Ambiguous-contact and invalid-contact classes were also removed from the local feature vocabulary. The current vocabulary contains seven directly interpretable geometric labels.

### 13.6 Group-Aware Dataset Export

Dataset export was changed from frame-level random splitting to acquisition-group splitting. Every frame from one video sequence remains in one split. The exporter records assignments in `split_manifest.json`, and the calibration script requires this manifest unless an explicit override is supplied.

### 13.7 Encoder Calibration

A temperature-scaling script was added. It loads the held-out validation split, verifies model-label ordering when a label sidecar exists, fits a positive scalar temperature by validation NLL, reports NLL, ECE, and Brier score before and after calibration, and saves the result beside the encoder weights.

### 13.8 Runtime Probability Calibration

All relevant prediction paths apply the encoder temperature before softmax. Both the normal multi-touch interface and the algorithm-comparison interface use calibrated feature probabilities when the sidecar exists.

### 13.9 Conservative Reliability Weighting

The set accumulator now supports entropy-derived touch reliability. This mechanism is enabled only when encoder calibration has been loaded. Without calibration, the registry passes unit weight so that low raw softmax entropy is not incorrectly equated with reliability.

### 13.10 Learned RFS Templates

A template-fitting script was added for object-level trials. It learns:

- Dirichlet-smoothed shape feature-frequency templates;
- a global feature prior;
- Beta-Bernoulli trial-level presence probabilities;
- a uniform experimental class prior.

The fitted configuration is saved as a model-specific RFS sidecar and automatically loaded by the prediction application.

### 13.11 Corrected Coverage Model

The implemented coverage model uses prior-corrected maximum coverage. It does not use noisy-OR accumulation, so many weak predictions cannot produce false certainty that a feature appeared. It also avoids treating an unobserved expected feature as evidence against a class.

### 13.12 Normalized Evidence

Semantic compatibility is divided by effective touch count. This prevents its scale from increasing with trial length and prevents the relative effect of evidence and coverage coefficients from changing solely because more contacts were collected.

### 13.13 Explicit Uncertain Outcome

The stopping rule distinguishes confidence-based acceptance from maximum-touch termination. If the touch budget is exhausted without satisfying the acceptance conditions, the result is marked uncertain while retaining the best current hypothesis for analysis.

### 13.14 Validation-Only Stopping Tuner

A stopping-parameter tuner evaluates threshold combinations on validation trials. It repeatedly permutes each trial to measure acquisition-order sensitivity and reports overall accuracy, accepted accuracy, acceptance rate, uncertain rate, mean touches, effective touches, and accuracy by touch count. An optional independent test file is evaluated only after parameter selection.

### 13.15 Automated Verification

Automated tests verify configuration loading, registry dispatch, Bayesian artifacts, grouped dataset splitting, encoder calibration mathematics, RFS trial persistence, encoder-label alias conversion, RFS template fitting, semantic compatibility equations, normalized beliefs, maximum coverage behavior, permutation invariance, stopping decisions, reliability weighting, and evaluation metrics. At the time this methodology document was prepared, the complete test suite contained 53 passing tests.

## 14. Expected Benefits and Research Significance

The principal expected benefit is improved recognition from ambiguous local observations. The method can retain weak evidence for several possible features rather than committing immediately to one. It can also use a rare distinctive feature without requiring every trial to observe every expected feature.

A second expected benefit is robustness to variable touch count. Normalized compatibility permits trials of different lengths to be compared without allowing score magnitude to grow automatically with the number of contacts.

A third expected benefit is resistance to repeated-contact overconfidence. Maximum coverage prevents repeated detection of the same local geometry from masquerading as evidence diversity.

A fourth expected benefit is efficient interaction. The system can stop when sufficient reliable and stable evidence has been obtained, while returning uncertain when the available contacts do not justify acceptance.

A fifth benefit is interpretability. Every object decision can be inspected through local feature distributions, feature coverage, frequency compatibility, unexpected evidence, class scores, confidence, entropy, and stopping reason. This is particularly useful in a dissertation because improvements or failures can be related to geometric evidence rather than only to a latent neural representation.

The contribution is the integration of calibrated local tactile semantics, permutation-invariant variable-cardinality fusion, diversity-aware feature coverage, trial-level presence modeling, and uncertainty-based stopping for handheld visuo-tactile primitive-shape recognition. Until a systematic literature review and experimental evaluation are complete, this should be presented as an integrated methodological contribution and research hypothesis rather than a claim that no related system has ever existed.

## 15. Limitations and Validity Conditions

### 15.1 Identifiability

The method cannot recover information that is absent from all local contacts. Objects distinguished primarily by global dimensions, such as a cube and a rectangular prism, may be locally indistinguishable when touch position, displacement, object dimensions, and proprioception are unavailable. This is why rectangular prism is excluded from the initial pose-free class set.

### 15.2 Touch-Policy Dependence

Both frequency and presence templates depend on exploration policy. Training and evaluation must use compatible policies, or policy variation must be represented in the dataset. User-held-out evaluation is necessary to measure robustness to different exploration behavior.

### 15.3 Correlated Contacts

Normalized evidence and maximum coverage reduce duplicate inflation but do not explicitly estimate contact correlation or detect that two contacts came from the same physical location. A future extension could incorporate embedding similarity, contact displacement, or duplicate-contact detection.

### 15.4 Calibration Shift

Temperature calibration is valid only to the extent that the validation distribution matches deployment. Changes in sensor illumination, gel condition, contact force, object material, or operator behavior may degrade calibration.

### 15.5 Semantic Vocabulary Dependence

The method depends on a useful local feature vocabulary. If two global classes produce indistinguishable distributions over the chosen features, more data will not resolve the problem. Additional local semantics or global sensing information would be required.

### 15.6 RFS Terminology

The present algorithm does not estimate a formal cardinality distribution, multi-object state, PHD, or labeled RFS posterior. Referring to it simply as a full RFS filter would overstate the mathematical machinery. The most accurate name is **uncertainty-aware set-based evidence accumulation**, with RFS presented as the representation principle motivating unordered variable-cardinality observations.

## 16. Research Hypotheses

The primary research hypothesis is:

> Primitive object shape can be recognized more accurately and efficiently by accumulating calibrated uncertain local tactile semantics from an unordered collection of unregistered contacts than by applying sequential Bayesian updates to hard local labels.

Supporting hypotheses are:

1. Full local feature distributions will outperform hard top-feature labels.
2. Maximum semantic coverage will improve discrimination between shapes that share common local surfaces.
3. Normalized evidence will reduce overconfidence caused by repeated contacts.
4. Calibrated entropy weighting will improve accepted-prediction reliability, although this must be confirmed by ablation.
5. Confidence-, entropy-, evidence-, and stability-based stopping will reduce mean touch count while preserving a target accepted accuracy.
6. Learned frequency and presence templates will outperform manually specified templates when sufficient independent object-level trials are available.

## 17. Methodological Summary

The baseline Bayesian classifier multiplies hard feature likelihoods across touches. It is simple and interpretable, but loses local uncertainty, treats repeated correlated observations as accumulating independent evidence, and does not model complementary feature coverage or explicit rejection.

The proposed method represents each trial as an unordered finite collection of calibrated local semantic distributions. It calculates reliability-weighted average compatibility with each shape, tracks prior-corrected maximum coverage of distinct local geometries, rewards expected observed features, penalizes unexpected observed features, and avoids penalizing expected features that were never sampled. The resulting shape scores are normalized into object beliefs and evaluated with an uncertainty-aware stopping rule.

The methodology is supported by group-aware dataset splitting, encoder temperature calibration, data-driven frequency and presence estimation, model-specific dynamic configuration, repeated permutation evaluation, validation-only stopping optimization, and automated mathematical tests. Together, these components establish a reproducible framework for testing whether unordered multi-touch semantic evidence can support efficient handheld visuo-tactile shape recognition.
