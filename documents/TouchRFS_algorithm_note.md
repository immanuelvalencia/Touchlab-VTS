# Uncertainty-Aware Set-Based Evidence Accumulation for Handheld Visuo-Tactile Shape Recognition

## Abstract

Handheld visuo-tactile sensing provides high-resolution local contact observations, but each contact captures only a small portion of an object's surface. This locality makes primitive shape recognition nontrivial: a planar contact may correspond to a cube face or a pyramid base, while a curved contact may correspond to a sphere, cylinder, or cone. This paper proposes an uncertainty-aware set-based evidence accumulation framework for primitive shape recognition from unregistered handheld GelSight contacts. Each tactile image is first interpreted as a probability distribution over local geometric tactile features: planar, single curvature, double curvature, straight edge, circular rim, multi-face vertex, or sharp apex. Cone and cylinder sides share the single-curvature label because their difference may not be identifiable from a small local contact. These uncertain semantic observations are then accumulated as an unordered variable-length set to infer the object class. The method is designed for pose-free tactile exploration, where contact positions and sensor orientations are unavailable. The framework explicitly models local feature uncertainty, positive feature coverage, object belief, and confidence-based early stopping. The proposed contribution is the integration of semantic tactile feature recognition, permutation-invariant evidence accumulation, and uncertainty-aware stopping for handheld primitive shape recognition.

## Introduction

### Background

Visuo-tactile sensors such as GelSight capture contact information by imaging the deformation of a soft elastomer. The resulting tactile images contain rich local geometry and texture cues, making them suitable for computer-vision methods such as convolutional neural networks. These sensors are especially useful when visual perception is unreliable, when objects are occluded, or when local geometry must be measured through contact.

Tactile object recognition differs fundamentally from visual object recognition. A visual image can often contain the full object, while a tactile contact typically contains only a local surface patch. Consequently, the information available from a single tactile image may be insufficient to identify the object. Primitive shapes illustrate this problem clearly. A cube and square pyramid may both produce planar contacts; a sphere, cylinder, and cone may all produce curved contacts; and cones and pyramids may both produce point-like contacts.

This work studies primitive shape recognition using a handheld GelSight Mini sensor. The task is to classify object shape from a variable number of manually collected tactile contacts, without assuming known contact pose, known touch order, robot kinematics, or registered surface locations.

### Problem Statement

Let a tactile trial consist of a finite collection of tactile images collected from one unknown object:

$$
X_T = \{I_1, I_2, \ldots, I_T\}
$$

The number of touches may vary between trials. The order of touches is not assumed to encode a controlled exploration trajectory. Each tactile image provides uncertain local information about the object surface.

The objective is to estimate the object class:

$$
\hat{C}_T \in \mathcal{C}
$$

from the unregistered tactile set:

$$
X_T
$$

where the candidate class set for the pose-free study is:

$$
\mathcal{C}
=
\{\mathrm{cube},\mathrm{sphere},\mathrm{cylinder},\mathrm{cone},\mathrm{square\ pyramid}\}
$$

Rectangular prism is excluded from the first pose-free study because it is generally not identifiable from unregistered local features alone. A cube and a rectangular prism can both produce planar faces, straight edges, and three-face vertices. Their distinction depends primarily on global dimensions and spatial arrangement, which are deliberately unavailable in the pose-free setting.

### Research Gap

Existing tactile object recognition methods demonstrate that multiple contacts can improve recognition, but many methods either use engineered descriptors, require spatial or proprioceptive information, operate on fixed-length representations, or fuse object predictions directly without preserving uncertainty over local tactile features.

The gap addressed in this work is:

> Primitive shape recognition from an unordered variable-length set of unregistered handheld tactile contacts, using explicit uncertainty over local geometric tactile semantics.

The emphasis is not on tracking distinct physical faces, edges, or vertices. Without contact pose, repeated observations of an edge cannot be reliably assigned to the same physical edge or to different edges. Instead, the proposed method treats each contact as an uncertain semantic observation and accumulates evidence over feature types.

### Proposed Approach

The proposed approach has two stages. First, a local tactile feature recognizer maps each GelSight image to a probability distribution over geometric tactile features. Second, a set-based evidence accumulation model combines these local feature distributions into an object-level belief.

The feature vocabulary is designed to correspond to meaningful local geometry:

- planar
- single curvature
- double curvature
- straight edge
- circular rim
- multi-face vertex
- sharp apex

This vocabulary can be interpreted through approximate principal curvature:

| Local geometry | Approximate interpretation |
|---|---|
| Planar | Both principal curvatures near zero. |
| Single curvature | One dominant curvature and one near-zero curvature, as on a cylinder or cone side. |
| Double curvature | Two positive curvatures, as on a sphere. |
| Edge | Curvature or surface-normal discontinuity. |
| Vertex or apex | Multiple discontinuities meeting in a small region. |

The CNN does not need to analytically compute curvature. The purpose of the vocabulary is to make local labels geometrically meaningful and discriminative for primitive shapes.

### Contributions

The proposed contribution is the integration of:

1. A curvature-aware semantic feature vocabulary for primitive-shape tactile discrimination.
2. A two-stage tactile recognition framework that separates local contact interpretation from global object classification.
3. A set-based evidence accumulation model for unordered, variable-length, unregistered tactile contacts.
4. A positive feature coverage score that rewards observed complementary evidence without treating unobserved features as absent.
5. An uncertainty-aware stopping rule based on confidence, entropy, and prediction stability.
6. An interpretable classification record showing which local tactile features support the final shape prediction.

## Related Literature

### GelSight and Visuo-Tactile Recognition

GelSight sensors have been widely used for high-resolution tactile perception because they convert contact deformation into image-like measurements. This enables the use of standard computer-vision pipelines while preserving tactile information about local geometry and surface structure.

Lin, Calandra, and Levine studied touch-based instance recognition using two robot-mounted GelSight sensors and visual object observations. Their work demonstrates that GelSight touch can support object recognition at scale, including recognition across 98 objects. However, the task is framed as visual-tactile matching with robot-mounted sensors, whereas the present work addresses tactile-only primitive shape classification from handheld unregistered contacts. Source: [Learning to Identify Object Instances by Touch](https://arxiv.org/abs/1903.03591).

Human-held GelSight Mini data collection has also been explored for material classification. Shuvo et al. collected GelSight Mini videos across multiple regions and pressing conditions and applied feature extraction and transfer learning to classify material categories. This work is relevant because it uses human-held GelSight Mini acquisition, but its target is material classification rather than primitive shape recognition from semantic local features. Source: [Material Classification using Visio-Tactile Sensor for Haptic Feedback Generation](https://pure.ulster.ac.uk/en/publications/material-classification-using-visio-tactile-sensor-for-haptic-fee/).

These studies establish the feasibility of GelSight-based classification. The present work differs by focusing on the structure of multi-contact tactile evidence rather than on single-frame or keyframe classification alone.

### Multi-Contact Tactile Shape Recognition

Multi-contact tactile recognition is a natural approach because each tactile observation is local. Tactile-SIFT extracts local descriptors from tactile images and uses a bag-of-words representation for object recognition. This demonstrates that descriptors from multiple contacts can support shape classification. However, bag-of-words aggregation produces a fixed-length descriptor and does not explicitly maintain uncertainty over semantic local features. Source: [Novel Tactile-SIFT Descriptor for Object Shape Recognition](https://livrepository.liverpool.ac.uk/3017790/).

iCLAP combines tactile appearance with proprioceptive information by linking tactile feature labels to 3D contact positions. This allows shape recognition to exploit both local tactile features and spatial arrangement. The method is highly relevant to primitive shape recognition, but it assumes spatial contact information. The present work addresses the pose-free handheld case where contact locations and sensor orientations are not assumed. Source: [iCLAP: shape recognition by combining proprioception and touch sensing](https://link.springer.com/article/10.1007/s10514-018-9777-7).

Active tactile recognition methods select actions that reduce recognition uncertainty. Monte Carlo Tree Search has been used to choose tactile exploration actions for object recognition. Such methods address where to touch next, whereas the present work addresses how to classify from an unordered set of contacts already obtained through handheld exploration. Source: [Active Tactile Object Recognition by Monte Carlo Tree Search](https://www.researchgate.net/publication/314152759_Active_Tactile_Object_Recognition_by_Monte_Carlo_Tree_Search).

### Set-Based Learning

Deep Sets introduced a general architecture for permutation-invariant learning over sets. Since unordered tactile contacts should not depend on presentation order, Deep Sets is an important baseline for the proposed method. Source: [Deep Sets](https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html).

Set Transformer extends set learning with attention mechanisms, allowing interactions among set elements while preserving permutation invariance. It is a strong baseline when sufficient training data are available. Source: [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html).

The proposed framework differs from generic set networks by introducing an explicit semantic intermediate representation. Rather than directly learning a black-box set-to-class mapping, the method accumulates interpretable probabilities over local tactile features and compares them with shape-level evidence templates.

### Random Finite Sets

Random finite set theory provides a mathematical framework for unordered sets with uncertain cardinality. It is widely used in multi-object tracking, mapping, SLAM, and sensor fusion, where the number of targets or landmarks is unknown. Source: [Random Finite Sets for Robot Mapping and SLAM](https://link.springer.com/book/10.1007/978-3-642-21390-8).

The tactile setting considered here shares the unordered and variable-cardinality structure of random finite set problems. However, full random finite set filters such as PHD, CPHD, LMB, and GLMB are designed for estimating multiple physical targets or features and require measurement and association models. In unregistered handheld tactile shape recognition, the system does not know whether repeated observations of an edge correspond to one physical edge or multiple physical edges.

Accordingly, this work uses set-based observation modeling but does not formulate the problem as full multi-target random finite set filtering. The set elements are uncertain semantic tactile observations, not tracked physical object parts.

## Methodology

### Overview

The proposed framework consists of two stages:

1. Local tactile feature recognition.
2. Semantic set-based evidence accumulation.

The first stage transforms each GelSight image into a probability distribution over local tactile features. The second stage combines these distributions into an object belief over primitive shape classes.

### Notation

The object class is denoted by:

$$
C
$$

The candidate object class set is:

$$
\mathcal{C}
$$

The local tactile feature is denoted by:

$$
F
$$

The feature vocabulary is:

$$
\mathcal{F}
$$

The tactile image collected at touch t is:

$$
I_t
$$

The unordered set of tactile images after T touches is:

$$
X_T = \{I_1, I_2, \ldots, I_T\}
$$

The predicted object class after T touches is:

$$
\hat{C}_T
$$

### Local Tactile Feature Recognition

A tactile feature recognizer estimates the local semantic content of each contact image. For each touch image, the model outputs:

$$
q_t(f) = P(F_t = f \mid I_t)
$$

The feature probabilities satisfy:

$$
\sum_{f \in \mathcal{F}} q_t(f) = 1
$$

where each value represents the probability that the current contact belongs to feature type f.

The output is a posterior distribution over local features, not a likelihood of the tactile image. This distinction matters because the later evidence score is a semantic compatibility score rather than a generative image likelihood.

### Contact Validity and Feature Confidence

Some contacts may be weak, saturated, partial, near the sensor boundary, or otherwise uninformative. The model can estimate valid-contact probability:

$$
r_t = P(V_t = 1 \mid I_t)
$$

where r_t lies between zero and one.

Feature ambiguity can be measured using entropy:

$$
H_F(t)
=
-\sum_{\substack{f \in \mathcal{F} \\ q_t(f)>0}}
q_t(f)\log q_t(f)
$$

The standard convention is that zero times the logarithm of zero equals zero. This is the exact Shannon entropy in natural-log units, or nats.

A normalized feature confidence score is:

$$
u_t
=
1
-
\frac{H_F(t)}{\log|\mathcal{F}|}
$$

One possible touch weight is:

$$
w_t = r_t u_t
$$

However, low entropy does not guarantee correctness because a neural network can be confidently wrong. Therefore, the feature classifier should be calibrated using a held-out validation set, temperature scaling, deep ensembles, or Monte Carlo dropout before entropy confidence is used as a weight.

The safer first weighting scheme is:

$$
w_t = r_t
$$

The entropy-weighted version should be evaluated as an ablation:

$$
w_t = r_t u_t
$$

If no validity or confidence weighting is used, set:

$$
w_t = 1
$$

### Shape Feature Templates

Each object class is represented by an expected distribution over local tactile features:

$$
\theta_c(f) = P(F = f \mid C = c,\pi)
$$

The symbol pi denotes the contact policy or user exploration behavior. This dependence is important: the features observed from a cube depend not only on the cube, but also on whether the user mainly touches faces, edges, or corners.

The template satisfies:

$$
\sum_{f \in \mathcal{F}} \theta_c(f) = 1
$$

For a cube, the template should assign high probability to planar, straight-edge, and multi-face-vertex contacts. For a sphere, the template should assign high probability to double-curvature contacts. Cylinder and cone sides should both contribute single-curvature contacts. Circular rims support both classes, while a sharp-apex contact provides the strongest local evidence favoring a cone.

The contact protocol should therefore be specified clearly. For handheld data, "unregistered contacts" or "uncontrolled contacts" is more accurate than "random touches" unless contact locations are randomized by a defined procedure.

### Learning the Feature Template From Data

If training trials are available, estimate the feature distribution for each object class by averaging predicted feature probabilities across touches. Reliability weights can be included so that weak contacts contribute less.

For class c:

$$
\theta_c(f)
=
\frac{
\alpha + \sum_{j:y_j=c}\sum_{t=1}^{T_j} w_{j,t}q_{j,t}(f)
}{
|\mathcal{F}|\alpha + \sum_{j:y_j=c}\sum_{t=1}^{T_j} w_{j,t}
}
$$

Here, y_j is the object label of training trial j, T_j is the number of touches in that trial, and alpha is a positive Dirichlet smoothing constant. The denominator normalizes the template so that the feature probabilities sum to one. When every weight equals one, this reduces to the unweighted estimator.

### Per-Touch Semantic Compatibility

The compatibility between touch t and class c is computed as the expected log template score under the predicted feature distribution:

$$
E_t^{\log}(c)
=
\sum_{f \in \mathcal{F}}
q_t(f)
\log\left(\theta_c(f)+\epsilon\right)
$$

This is an uncertainty-preserving cross-entropy compatibility score. It is not assumed to be the generative likelihood of the tactile image.

An alternative, more forgiving mixture compatibility score is:

$$
E_t^{\mathrm{mix}}(c)
=
\log\left(
\sum_{f \in \mathcal{F}}
q_t(f)\theta_c(f)
+
\epsilon
\right)
$$

The expected-log score penalizes template disagreement more strongly. However, a uniform local prediction can still favor a broad shape template under expected-log compatibility. The mixture score is therefore the primary implementation: when the local feature distribution is uniform and every shape template is normalized, it contributes the same semantic compatibility to every class. Expected-log compatibility remains an ablation to test whether stricter disagreement penalties improve validation performance.

### Normalized Set Evidence Score

Because summed evidence grows with the number of touches, the feature evidence score is normalized by the effective amount of evidence.

The effective evidence count is:

$$
N_{\mathrm{eff}}(T)
=
\sum_{t=1}^{T} w_t
$$

The normalized feature compatibility is:

$$
\bar{S}_{\mathrm{feat}}(c; X_T)
=
\frac{
\sum_{t=1}^{T} w_t E_t(c)
}{
N_{\mathrm{eff}}(T)+\epsilon
}
$$

Here, E_t(c) can be either the expected-log compatibility score or the mixture compatibility score.

This normalization prevents the semantic score from growing only because more contacts were collected. The amount of evidence is retained separately through the effective evidence count.

### Feature Coverage

Repeated weak probabilities for a feature should not accumulate into false certainty. A completely diffuse local prediction should also produce no coverage. Let the feature baseline be:

$$
\phi(f) = P(F=f)
$$

The baseline should be estimated from the local model's training or calibration distribution. If the local feature classes are balanced, the default is:

$$
\phi(f)=\frac{1}{|\mathcal{F}|}
$$

Define above-baseline feature support as:

$$
\widetilde{q}_t(f)
=
\max\left(
\frac{q_t(f)-\phi(f)}{1-\phi(f)},
0
\right)
$$

Feature coverage is then defined using maximum observed support:

$$
m_T(f)
=
\max_{1 \le t \le T}
w_t \widetilde{q}_t(f)
$$

The value m_T(f) is high only if at least one valid touch supports feature f above its baseline rate. A baseline-level prediction contributes zero, while certainty maps to one. The maximum prevents repeated weak probabilities from accumulating as they would under noisy-OR coverage.

This quantity measures semantic feature presence, not distinct physical part identity.

### Shape Presence Templates

Some features are expected to be useful evidence for each shape. This is represented by:

$$
a_c(f) \in [0,1]
$$

Unlike theta_c, this template does not need to sum to one. Multiple features can be expected for the same shape. For example, a cube may expect planar, straight-edge, and multi-face-vertex contacts simultaneously.

### Positive and Unexpected Coverage Scores

Under unregistered contacts, absence of a feature should not be treated as strong negative evidence. If a cone trial does not include an apex touch, that does not prove the object is not a cone. It may simply mean the apex was not sampled.

Therefore, the coverage score uses positive evidence from observed features:

$$
S_{\mathrm{cov}}^{+}(c; X_T)
=
\sum_{f \in \mathcal{F}}
a_c(f)m_T(f)
$$

Unexpected observed features can be penalized separately:

$$
S_{\mathrm{unexpected}}(c; X_T)
=
\sum_{f \in \mathcal{F}}
\left(1-a_c(f)\right)m_T(f)
$$

The combined coverage score is:

$$
S_{\mathrm{cov}}(c; X_T)
=
S_{\mathrm{cov}}^{+}(c; X_T)
-
\gamma S_{\mathrm{unexpected}}(c; X_T)
$$

The coefficient gamma controls how strongly unexpected observed features are penalized. No term penalizes an expected feature solely because it has not yet been observed.

### Total Class Score

The class ranking score combines class prior, normalized semantic compatibility, and coverage evidence:

$$
S(c; X_T)
=
\log\left(P(C=c)+\epsilon\right)
+
\lambda_E \bar{S}_{\mathrm{feat}}(c; X_T)
+
\lambda_C S_{\mathrm{cov}}(c; X_T)
$$

The effective evidence count is retained as a separate confidence feature:

$$
N_{\mathrm{eff}}(T)
$$

This value is not included in the class ranking score because it is class-independent. Adding it equally to every class would not change the predicted class or the softmax belief. It can instead be used by a separate calibration model or stopping rule.

The predicted class is:

$$
\hat{C}_T
=
\arg\max_{c \in \mathcal{C}}
S(c; X_T)
$$

This total score is a discriminative score, not a fully normalized generative probability model.

### Object Belief and Calibration

The class scores are converted into an object belief distribution using temperature-scaled softmax:

$$
b_T(c)
=
\frac{
\exp\left(S(c; X_T)/\tau_{\mathrm{cal}}\right)
}{
\sum_{c' \in \mathcal{C}}
\exp\left(S(c'; X_T)/\tau_{\mathrm{cal}}\right)
}
$$

The calibration temperature is learned on a validation set. Calibration should be evaluated separately for different touch counts because confidence can change as evidence accumulates. Before the temperature and score weights are fitted and calibration is verified, these values should be described as normalized decision scores rather than empirical class probabilities.

### Object Uncertainty

Object-level uncertainty is measured by entropy:

$$
H_C(T)
=
-\sum_{\substack{c \in \mathcal{C} \\ b_T(c)>0}}
b_T(c)\log b_T(c)
$$

Low entropy indicates a confident object belief. High entropy indicates that multiple object classes remain plausible.

### Early Stopping

The model can stop tactile exploration when the belief is confident, uncertainty is low, the prediction is stable, and enough effective evidence has been collected.

Define confidence, entropy, and stability conditions:

$$
\max_{c \in \mathcal{C}} b_T(c) \ge \tau_{\mathrm{stop}}
$$

$$
H_C(T) \le \eta
$$

$$
\hat{C}_T
=
\hat{C}_{T-1}
=
\hat{C}_{T-2}
$$

$$
N_{\mathrm{eff}}(T) \ge \nu
$$

The stopping decision is:

$$
\mathrm{Stop}(T)
=
\begin{cases}
1, & \text{if confidence, entropy, stability, and evidence conditions hold} \\
1, & \text{if } T \ge T_{\max} \\
0, & \text{otherwise}
\end{cases}
$$

Stopping and accepting a class are distinct decisions. If all confidence criteria hold, the argmax class is accepted. If the maximum touch count is reached first, acquisition stops but the decision is returned as uncertain; the argmax class may still be reported as the best candidate for analysis.

Final classification for a fixed contact set is permutation-invariant. Early stopping time, however, can depend on the order in which contacts arrive because stopping is evaluated on prefixes of the contact set.

## Experimental Design

### Object Classes

The pose-free study should use primitive shapes that are distinguishable through local geometric feature combinations:

- cube
- sphere
- cylinder
- cone
- square pyramid

Rectangular prism should be excluded unless global geometric information is added, such as contact position, estimated face extent, known finger displacement, object dimensions, or proprioception.

### Local Feature Labels

The local feature vocabulary should be:

- planar
- single curvature
- double curvature
- straight edge
- circular rim
- multi-face vertex
- sharp apex

These labels train and evaluate the local tactile feature recognizer. The vocabulary is designed to distinguish sphere, cylinder, and cone more reliably than a generic curved-surface label.

### Collection Protocol

The contact protocol must be specified because the observed feature distribution depends on how users touch the object.

Possible protocols include:

- uncontrolled handheld contacts
- instructed diverse exploration
- touch any location without attempting to identify the shape
- seek different surface regions while avoiding repeated locations

The term random touches should be used only if contact locations are randomized by a defined procedure.

### Evaluation Splits

The primary split should be by physical object instance. Random image-level splitting is not sufficient because contacts from the same object instance may share texture, size, wear, or manufacturing artifacts.

Recommended splits:

| Split | Purpose |
|---|---|
| Trial-held-out | Tests new trials from seen object instances. |
| Object-instance-held-out | Tests generalization to unseen objects of the same primitive shape. |
| User-held-out | Tests robustness to different handheld touch behavior. |

The object-instance-held-out split should be treated as the main result.

### Baselines

The proposed method should be compared against:

- single-touch CNN object classification
- average softmax fusion across touches
- majority voting across touches
- Bayesian sequential fusion over object labels
- Deep Sets over local feature probabilities
- Deep Sets over learned tactile embeddings
- Set Transformer
- GRU or LSTM if touch order is recorded
- proposed method without coverage score
- proposed method without confidence weighting
- proposed method with expected-log compatibility
- proposed method with mixture compatibility
- proposed method without early stopping

### Metrics

The evaluation should report:

- accuracy as a function of touch count
- macro F1 score
- confusion matrix
- object entropy as a function of touch count
- expected calibration error
- average stopping time
- wrong early-decision rate
- uncertain or rejection rate
- performance under shuffled final contact sets
- distribution of stopping times under random contact-order permutations
- performance on unseen object instances

The most important curves are:

$$
\mathrm{Accuracy}(T)
$$

$$
H_C(T)
$$

$$
\mathrm{Average\ stopping\ time}
$$

### Ablation Study

The ablation study should isolate the contribution of each component:

| Variant | Purpose |
|---|---|
| No coverage score | Tests whether complementary feature presence improves classification. |
| No confidence weighting | Tests whether low-quality or uncertain contacts should be discounted. |
| Validity-only weighting | Tests w_t equals r_t against entropy-based weighting. |
| Entropy-based weighting | Tests whether calibrated feature confidence improves performance. |
| Hard local feature labels | Tests whether preserving feature uncertainty matters. |
| Expected-log compatibility | Tests the stronger penalty for template mismatch. |
| Mixture compatibility | Tests the more forgiving compatibility score. |
| Summed evidence | Tests whether unnormalized evidence improves accuracy but worsens calibration. |
| Normalized evidence | Tests whether evidence normalization stabilizes confidence across touch counts. |
| No early stopping | Tests classification without touch-efficiency optimization. |
| Deep Sets baseline | Tests against generic permutation-invariant fusion. |
| Set Transformer baseline | Tests against attention-based set modeling. |
| Shuffled final set | Confirms order-invariant final classification. |
| Permuted acquisition order | Measures how stopping time depends on touch arrival order. |

## Distinction From Alternative Methods

### Single-Touch CNN Classification

Single-touch CNN classifiers attempt to predict the object class from one tactile image. This is often underdetermined because one local contact may be shared by multiple object classes.

The proposed framework predicts local feature probabilities first and then accumulates evidence over multiple contacts.

### Averaging and Voting

Averaging and voting combine object-level predictions from individual touches. They do not explicitly model which local tactile features were observed or whether the observed feature set is sufficient to distinguish a shape.

The proposed method accumulates semantic feature compatibility and positive feature coverage.

### Bayesian Object Fusion

Bayesian fusion can sequentially update object class probabilities. However, object-level fusion may obscure the local tactile reason for each update.

The proposed framework accumulates local semantic features before object-level inference, improving interpretability.

### Deep Sets and Set Transformer

Deep Sets and Set Transformer provide generic set-learning architectures. They are appropriate baselines because the tactile contacts form an unordered set.

The proposed framework differs by imposing semantic structure. It explicitly models local feature probabilities, shape templates, feature coverage, object uncertainty, and stopping criteria.

### iCLAP

iCLAP combines tactile feature labels with contact positions. It is well suited to tactile-kinesthetic shape recognition.

The proposed framework is designed for pose-free handheld contacts, where contact position and orientation are not assumed.

### Tactile-SIFT

Tactile-SIFT uses engineered tactile descriptors and bag-of-words aggregation.

The proposed framework uses probabilistic semantic feature predictions and uncertainty-aware set-based accumulation.

### Full RFS Filters

Full random finite set filters estimate sets of physical targets or features and generally require measurement and association models.

The proposed framework does not estimate distinct physical faces, edges, or vertices. It accumulates semantic observations over feature types. This makes the method better suited to unregistered handheld tactile contacts.

## Conclusion

This paper proposes an uncertainty-aware set-based evidence accumulation framework for primitive shape recognition using handheld GelSight contacts. The method addresses the local and ambiguous nature of tactile sensing by separating local feature recognition from global object inference. Each touch contributes an uncertain semantic observation, and the object class is inferred from normalized semantic compatibility and positive feature coverage across an unordered variable-length contact set.

The research hypothesis is:

> Primitive object shape can be recognized efficiently by accumulating uncertain local tactile semantics from unregistered handheld contacts.

The framework is designed to be interpretable, pose-free, variable-length, and suitable for early stopping. Its contribution lies in the integration of curvature-aware local tactile semantics, uncertainty-preserving set-based accumulation, positive feature coverage, calibration-aware confidence, and stability-based stopping for handheld visuo-tactile primitive shape recognition.
