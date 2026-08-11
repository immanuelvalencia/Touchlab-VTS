# Speaker Script: Uncertainty-Aware Set-Based Evidence Accumulation for Handheld Visuo-Tactile Shape Recognition

## Opening

Good day everyone. Today I will explain the motivation, background, and proposed method for a visuo-tactile object recognition project using a GelSight Mini sensor.

The project focuses on a simple but important problem:

> How can a handheld tactile sensor recognize primitive object shapes from several local touches?

The target shapes are primitive three-dimensional objects such as:

- cube
- sphere
- cylinder
- cone
- square pyramid

At first, these objects may seem easy to classify. However, for tactile sensing, they are actually a useful and challenging starting point because touch is local.

A camera can often see the whole object. A tactile sensor cannot. A tactile sensor only feels the small region that is currently in contact with the sensor surface.

This means a single tactile image may not contain enough information to identify the object.

For example:

- A flat contact could come from the face of a cube or the base of a pyramid.
- A curved contact could come from a sphere, a cylinder, or a cone.
- A sharp point could come from a cone tip or a pyramid apex.

So the central idea of this project is:

> The object should not be classified from one tactile image alone. It should be classified by accumulating evidence from multiple local tactile impressions.

## What Is a Visuo-Tactile Sensor?

A visuo-tactile sensor is a tactile sensor that produces image-like data.

GelSight is a good example. It has a soft gel surface. When an object presses against the gel, the gel deforms. A camera inside the sensor observes this deformation. The result is an image that contains information about local contact geometry.

So although the sensor is measuring touch, the output looks like an image. This is useful because image-processing and deep-learning methods can be applied to tactile data.

However, GelSight images are not normal camera images. They do not show the entire object. They show local surface contact.

This is why a standard image classifier is not enough.

## The Main Problem

The problem is local ambiguity.

Suppose the sensor touches a flat patch. This is useful information, but it does not uniquely identify the object.

A flat patch may come from:

- a cube
- a square pyramid base
- a cylinder base
- a cone base

Suppose the sensor touches a curved patch. That may come from:

- a sphere
- a cylinder
- a cone

The local contact tells us something, but not everything.

Therefore, the project asks:

> Can we recognize the object by collecting several uncertain local touches and combining them intelligently?

## Why Not Just Use Object Labels Directly?

A simple approach would be to train a neural network that receives one GelSight image and directly predicts the object label:

```text
cube
sphere
cylinder
cone
pyramid
```

This is the most straightforward method, and many tactile classification systems begin with this idea. The sensor image is treated like a normal image, a CNN is trained, and the output is the object class.

This can work when the tactile image contains enough class-specific information. For example, if the object has a unique texture or a very distinctive local shape, then direct object classification may perform well.

But for primitive shape recognition, this approach has a serious limitation:

> the tactile image may show only a local feature, not the object.

If the sensor touches a flat face, the local image may be genuinely compatible with several objects. It could be a cube face, the base of a square pyramid, the base of a cylinder, or the base of a cone.

In that situation, asking the neural network to output one object label from one local contact can be physically unfair. The information needed to identify the object is not present in the image.

So if the network predicts "cube" from a flat patch, it may not be because the image truly proves cube. It may be because the training data taught the network a dataset bias.

For example, it may learn:

```text
flat patch with this lighting -> cube
flat patch with this indentation -> pyramid
flat patch from this object size -> cylinder
```

Those cues may not generalize to new objects, new sizes, new users, or new touch forces.

This is one of the main risks of direct object labeling:

> the network may learn object-instance artifacts instead of shape reasoning.

### Current Method 1: Single-Touch Object Classification

The first current method is single-touch classification.

The input is:

```text
one tactile image
```

The output is:

```text
object class
```

The limitation is that a single contact may be underdetermined.

For example:

```text
planar contact -> cube or pyramid or cylinder base or cone base
curved contact -> sphere or cylinder or cone
sharp contact -> cone or pyramid
```

So the classifier may be forced to make a decision before enough evidence exists.

### Current Method 2: Multi-Touch Voting or Averaging

A second common method is to classify each touch independently and then combine the predictions.

For example:

```text
Touch 1 -> cube: 0.45, pyramid: 0.35
Touch 2 -> cube: 0.40, pyramid: 0.38
Touch 3 -> cube: 0.50, pyramid: 0.30
Average -> cube
```

This is better than using only one touch, but it still has a limitation.

Each touch is forced to produce an object-level prediction, even if that touch only contains local information.

So the system is averaging object guesses, not accumulating tactile features.

Another problem is repeated contacts. If the user touches the same flat face five times, voting or averaging may treat those five touches as five separate confirmations. But physically, they may all be the same kind of evidence.

In other words:

```text
flat + flat + flat
```

should not be as informative as:

```text
flat + edge + vertex
```

### Current Method 3: Sequence Models

Another possible method is to use a sequence model such as an LSTM, GRU, or temporal Transformer.

These models process touches in order:

```text
touch 1 -> touch 2 -> touch 3 -> final class
```

This can be useful if the order of touches is meaningful. For example, in a robot-controlled exploration policy, the robot may deliberately touch one region, then move to another region, then follow an edge.

But in handheld tactile sensing, the order may not be meaningful. One user may touch the face first, another may touch the edge first, and another may touch the apex first.

For final object classification, the set of contacts may matter more than the order.

So a sequence model may learn accidental order patterns from the dataset.

### Current Method 4: Generic Set Models

A more suitable modern approach is to use a set model, such as Deep Sets or Set Transformer.

These models can accept multiple touches without caring about order.

This is a strong baseline.

However, generic set models can still be hard to interpret. They may tell us the final answer, but not clearly explain which tactile features caused the answer.

For this project, interpretability matters because the goal is not only to get high accuracy. The goal is to understand how local tactile evidence supports primitive shape recognition.

### Proposed Alternative

Instead of assigning object labels directly to every single touch, the proposed method separates the problem into two levels:

```text
single touch -> local tactile feature
multiple touches -> object shape
```

For one touch, the model predicts local features such as:

- planar
- single curvature
- double curvature
- straight edge
- circular rim
- multi-face vertex
- sharp apex

Then, after several touches, the system uses the pattern of local features to classify the object.

For example:

```text
planar + straight edge + multi-face vertex -> cube
```

or:

```text
single curvature + circular rim + sharp apex -> cone
```

or:

```text
planar + straight edge + sharp apex -> square pyramid
```

This is more physically meaningful because the model does not pretend that one local contact always identifies the whole object. Instead, it lets each touch say what it can reasonably say, then combines several touches into an object-level decision.

The main advantage is interpretability.

The system can explain:

```text
I predicted cube because I observed planar contacts, edges, and a vertex.
```

or:

```text
I predicted cone because I observed single curvature, a circular rim, and an apex. The apex distinguishes this combination from a cylinder.
```

This is the reason for using semantic local features before object classification.

## Why Some Shapes Are Excluded

The first study should not include rectangular prism in the pose-free setup.

The reason is information limitation.

A cube and a rectangular prism can both produce:

- planar faces
- straight edges
- three-face vertices
- right-angle geometry

The main difference is global dimension. A cube has equal side lengths. A rectangular prism has unequal side lengths.

But if the system does not know contact location, touch displacement, object dimensions, or global spatial arrangement, it may not have enough information to distinguish them.

So the first pose-free study should use:

$$
\mathcal{C}
=
\{\mathrm{cube},\mathrm{sphere},\mathrm{cylinder},\mathrm{cone},\mathrm{square\ pyramid}\}
$$

Rectangular prism can be added later if global information is available.

Examples of useful global information would be:

- touch location
- contact displacement
- finger movement distance
- object dimensions
- proprioception
- estimated face extent

## What Is Evidence Accumulation?

Evidence accumulation means updating the decision as more information arrives.

After one touch, the system may be uncertain.

After two touches, the system may become less uncertain.

After five touches, the system may have enough evidence to decide.

For example:

```text
Touch 1: planar
Possible objects: cube, pyramid, cylinder base, cone base

Touch 2: straight edge
Possible objects: cube, pyramid

Touch 3: sharp apex
Most likely object: square pyramid
```

The idea is similar to how humans identify objects by touch. One contact may be insufficient, but several contacts can reveal the object.

In this project, each touch contributes uncertain evidence. The system accumulates this evidence until the object belief becomes confident.

## What Does "Uncertainty-Aware" Mean?

Uncertainty-aware means the system does not treat every prediction as equally reliable.

A tactile image may be clear. For example, a strong edge contact may be easy to identify.

Another tactile image may be ambiguous. For example, a weak or partial contact may look partly flat and partly curved.

Instead of forcing a hard label, the local model outputs a probability distribution.

For one touch, the model may output:

| Local feature | Probability |
|---|---:|
| planar | 0.10 |
| straight edge | 0.65 |
| single curvature | 0.08 |
| double curvature | 0.08 |
| circular rim | 0.01 |
| multi-face vertex | 0.06 |
| sharp apex | 0.02 |

This means the model is most confident that the contact is an edge, but it still preserves uncertainty.

Mathematically:

$$
q_t(f) = P(F_t=f \mid I_t)
$$

Simple explanation:

- t is the touch number.
- I_t is the tactile image from touch t.
- f is one possible local feature.
- q_t(f) is the probability that touch t belongs to feature f.

The probabilities must sum to one:

$$
\sum_{f \in \mathcal{F}} q_t(f) = 1
$$

This is important because the system accumulates soft evidence, not hard labels.

## What Is a Set?

A set is a collection where order does not matter.

For example:

```text
{flat, edge, apex}
```

is the same set as:

```text
{apex, flat, edge}
```

In handheld tactile exploration, the order of touches is often not meaningful. One user may touch the edge first, while another user may touch the flat face first.

So the final classification should not depend on touch order.

The touch set after T touches is:

$$
X_T = \{I_1, I_2, \ldots, I_T\}
$$

The number of touches T can vary.

This is why a set-based method is a good fit.

## What Is a Random Finite Set?

A Random Finite Set, or RFS, is a mathematical object used to represent a set where:

1. The number of elements is uncertain.
2. The values of the elements are uncertain.
3. The order of the elements does not matter.

In robotics, RFS methods are often used for multi-object tracking.

For example, a robot may observe several objects with a camera or radar:

```text
object 1
object 2
object 3
```

But the robot may not know:

- how many objects are really present
- which detections are false alarms
- which objects were missed
- which measurement belongs to which object

RFS is useful because it represents all detections as a set.

An RFS-style state could look like:

$$
Y = \{y_1,y_2,\ldots,y_n\}
$$

where n is uncertain.

This is powerful in robotics for:

- multi-object tracking
- SLAM
- mapping
- sensor fusion
- target detection

## Why Not Use a Full RFS Filter Here?

A full RFS filter, such as PHD, CPHD, LMB, or GLMB, is designed to estimate sets of physical targets or features.

For this tactile project, that would mean estimating physical object parts such as:

$$
\{\mathrm{face}_1,\mathrm{edge}_1,\mathrm{edge}_2,\mathrm{vertex}_1\}
$$

But this is difficult without contact pose.

If the sensor touches an edge twice, we do not know whether it touched:

```text
the same edge twice
```

or:

```text
two different edges
```

Without contact location or orientation, physical feature association is not reliable.

So this project does not use a full RFS tracking filter.

Instead, it uses the useful part of the RFS idea:

> Treat the observations as an unordered variable-size set.

The set elements are not physical object parts. They are uncertain semantic observations from tactile images.

This is why the method is better described as:

> uncertainty-aware set-based evidence accumulation

rather than full RFS filtering.

## How RFS Thinking Still Helps

Even though the method does not use a full RFS filter, RFS thinking is still useful.

The tactile observations have three RFS-like properties:

1. The number of touches is variable.
2. The touch order should not affect the final classification.
3. Each touch observation is uncertain.

So the project borrows the set representation:

$$
X_T = \{I_1,I_2,\ldots,I_T\}
$$

and applies it to tactile evidence accumulation.

The difference is:

```text
RFS tracking:
  estimate physical targets from measurements

This project:
  estimate object class from uncertain tactile observations
```

## Local Feature Vocabulary

A strong feature vocabulary is important.

Using only labels such as flat, edge, and curved may be too weak. For example, sphere, cylinder, and cone can all produce curved contacts.

So the proposed vocabulary uses curvature-aware tactile features:

| Feature | Meaning |
|---|---|
| Planar | Locally flat surface. |
| Single curvature | One main curvature, like a cylinder or cone side. |
| Double curvature | Curved in two directions, like a sphere. |
| Straight edge | Discontinuity between surfaces. |
| Circular rim | Circular edge, such as the base of a cone or cylinder. |
| Multi-face vertex | Several faces meeting, such as cube corner. |
| Sharp apex | Point-like tip, such as cone or pyramid apex. |

The CNN does not need to explicitly calculate curvature. It only needs to learn these categories from labeled examples.

The vocabulary is designed to make primitive shapes more distinguishable.

## Shape Templates

Each shape can be described by the local features it tends to produce.

For example:

```text
cube:
  planar
  straight edge
  multi-face vertex

sphere:
  double curvature

cylinder:
  single curvature
  circular rim
  planar base

cone:
  single curvature
  circular rim
  sharp apex

square pyramid:
  planar
  straight edge
  sharp apex
```

Mathematically, each class has a feature template:

$$
\theta_c(f) = P(F=f \mid C=c,\pi)
$$

Simple explanation:

- c is the object class.
- f is the local tactile feature.
- theta_c(f) says how often feature f is expected for class c.
- pi represents the contact policy, or how the user chooses touches.

The contact policy matters. If one user mainly touches flat faces and another user tries to touch corners, the observed feature distribution will differ.

This is why the experiment must define the collection protocol clearly.

The template must sum to one:

$$
\sum_{f \in \mathcal{F}} \theta_c(f) = 1
$$

## Semantic Compatibility

For each touch, the model compares the predicted feature probabilities with each shape template.

The expected-log compatibility score is:

$$
E_t^{\log}(c)
=
\sum_{f \in \mathcal{F}}
q_t(f)
\log\left(\theta_c(f)+\epsilon\right)
$$

Simple explanation:

- q_t(f) says what the touch probably is.
- theta_c(f) says what class c expects.
- If the touch matches the shape template, the score is better.
- The small epsilon avoids taking the log of zero.

This is not a true image likelihood. It is a semantic compatibility score.

Another option is the mixture compatibility score:

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

The expected-log score is stricter, but a completely uniform local prediction can still favor a broad shape template. The mixture score is the primary version because a uniform local prediction gives equal semantic compatibility to every normalized shape template. Expected-log compatibility remains useful as an experimental comparison.

## Normalized Evidence Accumulation

If evidence is simply summed, classes may become overconfident as more touches are added.

So the method uses normalized evidence.

First define the effective number of useful touches:

$$
N_{\mathrm{eff}}(T)
=
\sum_{t=1}^{T} w_t
$$

Here, w_t is the reliability weight of touch t.

Then compute normalized feature evidence:

$$
\bar{S}_{\mathrm{feat}}(c; X_T)
=
\frac{
\sum_{t=1}^{T} w_t E_t(c)
}{
N_{\mathrm{eff}}(T)+\epsilon
}
$$

Simple explanation:

- Each touch gives a compatibility score.
- Reliable touches count more.
- The total is divided by the effective number of useful touches.
- This prevents the score from growing only because more touches were collected.

## Feature Coverage

Evidence compatibility is useful, but the method also needs to know which feature types have appeared.

For example, for a cube, seeing:

```text
planar + straight edge + vertex
```

is more informative than seeing:

```text
planar + planar + planar
```

Feature coverage measures whether a feature has appeared confidently at least once. It should not claim that every feature appeared merely because an uncertain network assigned a small probability to every label.

First define a baseline probability for each local feature:

$$
\phi(f)=P(F=f)
$$

If the seven local labels are balanced, the default baseline is one seventh. Support above this baseline is:

$$
\widetilde{q}_t(f)
=
\max\left(
\frac{q_t(f)-\phi(f)}{1-\phi(f)},
0
\right)
$$

A baseline-level prediction becomes zero support. A probability of one becomes full support.

The proposed coverage is:

$$
m_T(f)
=
\max_{1 \le t \le T}
w_t \widetilde{q}_t(f)
$$

Simple explanation:

- If any valid touch strongly detects feature f, coverage is high.
- If no touch strongly detects feature f, coverage is low.
- Weak repeated predictions do not accumulate into false certainty.

This is safer than noisy-OR coverage for the first implementation.

## Positive Coverage

A key design choice is that absence should not be treated as strong negative evidence.

For example, if the object is a cone but the user has not touched the apex, the model should not strongly reject cone. The apex may simply not have been sampled.

So the method rewards observed expected features.

The positive coverage score is:

$$
S_{\mathrm{cov}}^{+}(c; X_T)
=
\sum_{f \in \mathcal{F}}
a_c(f)m_T(f)
$$

Here, a_c(f) says whether feature f is expected or useful for class c.

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

Simple explanation:

- Expected observed features help the class.
- Unexpected observed features can hurt the class.
- Missing expected features are not strongly punished.

This is important for unregistered handheld touch.

## Final Object Score

The final score for each class is:

$$
S(c; X_T)
=
\log\left(P(C=c)+\epsilon\right)
+
\lambda_E \bar{S}_{\mathrm{feat}}(c; X_T)
+
\lambda_C S_{\mathrm{cov}}(c; X_T)
$$

Simple explanation:

- The first term is the class prior.
- The second term measures average semantic compatibility.
- The third term measures feature coverage.
- The lambda values control how much each part matters.

The predicted class is:

$$
\hat{C}_T
=
\arg\max_{c \in \mathcal{C}}
S(c; X_T)
$$

This is a discriminative score, not a full generative probability model.

## Object Belief

The score can be converted into a belief distribution using softmax:

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

Simple explanation:

- b_T(c) is the belief that the object belongs to class c.
- tau_cal is a calibration temperature.
- Calibration helps prevent overconfident predictions.

The beliefs sum to one across classes.

Until the score weights and calibration temperature have been fitted on held-out trials, these numbers are normalized decision scores. They should only be interpreted as calibrated class probabilities after calibration has been measured and verified.

## Object Uncertainty

Object uncertainty can be measured using entropy:

$$
H_C(T)
=
-\sum_{\substack{c \in \mathcal{C} \\ b_T(c)>0}}
b_T(c)\log b_T(c)
$$

Simple explanation:

- Low entropy means the model is confident.
- High entropy means several classes are still plausible.

As more useful touches are collected, entropy should generally decrease.

## Early Stopping

The system should stop touching once it has enough evidence.

A possible stopping rule uses four conditions:

1. Confidence is high.
2. Entropy is low.
3. Prediction is stable.
4. Enough effective evidence has been collected.

Confidence condition:

$$
\max_{c \in \mathcal{C}} b_T(c) \ge \tau_{\mathrm{stop}}
$$

Entropy condition:

$$
H_C(T) \le \eta
$$

Stability condition:

$$
\hat{C}_T
=
\hat{C}_{T-1}
=
\hat{C}_{T-2}
$$

Evidence condition:

$$
N_{\mathrm{eff}}(T) \ge \nu
$$

Stopping rule:

$$
\mathrm{Stop}(T)
=
\begin{cases}
1, & \text{if all stopping conditions hold} \\
1, & \text{if } T \ge T_{\max} \\
0, & \text{otherwise}
\end{cases}
$$

Stopping acquisition and accepting a class are separate decisions. If all confidence conditions hold, the predicted class is accepted. If the maximum number of touches is reached first, acquisition stops but the result is marked uncertain. The highest-scoring class can still be shown as the best candidate without presenting it as a confident decision.

One important point:

> Final classification for a fixed set of touches is order-invariant, but stopping time can depend on the order in which touches arrive.

This is not a mistake. Early stopping operates on prefixes of the collected sequence.

## How This Differs From Other Studies

### Difference From Single-Touch CNNs

Single-touch CNNs try to classify the whole object from one local tactile image.

This project does not assume one touch is enough. It first classifies the local feature, then accumulates evidence across touches.

### Difference From Voting and Averaging

Voting and averaging combine object predictions from multiple touches.

This project combines semantic tactile features instead. It can explain the decision as:

```text
planar + edge + vertex -> cube
```

or:

```text
single curvature + rim + apex -> cone
```

### Difference From Deep Sets and Set Transformer

Deep Sets and Set Transformer are general set-learning models. They are strong baselines.

This project is more structured and interpretable. It explicitly models:

- local tactile feature probabilities
- feature templates
- feature coverage
- uncertainty
- early stopping

### Difference From iCLAP

iCLAP uses contact position and proprioception. That is powerful for shape recognition.

This project studies the harder pose-free handheld case, where contact location and orientation are not assumed.

### Difference From Full RFS Filters

Full RFS filters track physical targets or features.

This project does not track physical faces or edges. It accumulates semantic feature evidence.

So RFS is used as inspiration for set representation, not as a full tracking algorithm.

## Why This Study Is Needed

This study is needed because handheld tactile recognition has a different structure from robot-controlled tactile recognition.

In robot-controlled settings, the system may know:

- where the sensor touched
- how it moved
- what the contact pose was
- how different touches relate spatially

In handheld settings, this information may be unavailable.

Therefore, the algorithm must work with:

- unregistered contacts
- variable touch counts
- ambiguous local features
- uncertain predictions
- no global spatial map

The proposed method is designed for exactly this setting.

## Research Gaps

The main gaps are:

1. Many tactile classifiers operate on single images or fixed representations.
2. Multi-touch methods often require pose, proprioception, or spatial registration.
3. Generic set models are powerful but less interpretable.
4. Full RFS filters require association models that are not available in pose-free handheld touch.
5. Existing methods do not fully exploit uncertainty-preserving semantic local feature accumulation for primitive shape recognition.

This project addresses these gaps by using an interpretable set-based evidence model.

## Experimental Plan

The experiment should evaluate whether the method recognizes primitive shapes efficiently and reliably.

Object classes:

$$
\mathcal{C}
=
\{\mathrm{cube},\mathrm{sphere},\mathrm{cylinder},\mathrm{cone},\mathrm{square\ pyramid}\}
$$

Rectangular prism should be excluded unless global geometric information is added.

Important baselines:

- single-touch CNN
- average softmax fusion
- majority voting
- Bayesian object fusion
- Deep Sets
- Set Transformer
- proposed method without coverage
- proposed method without uncertainty weighting
- proposed method without early stopping

Important metrics:

- accuracy versus number of touches
- macro F1 score
- confusion matrix
- entropy versus number of touches
- expected calibration error
- average stopping time
- wrong early-decision rate
- uncertain decision rate
- user-held-out accuracy
- object-instance-held-out accuracy

The most important evaluation is object-instance-held-out testing. This checks whether the model learns shape rather than memorizing a specific object.

## Significance

The significance of this project is that it provides a principled way to classify objects from local tactile contacts without requiring robot pose or global reconstruction.

The method is significant because it is:

- pose-free
- variable-length
- uncertainty-aware
- interpretable
- suitable for early stopping
- designed for handheld visuo-tactile sensing

It also creates a bridge between RFS-inspired set thinking and practical tactile recognition.

The project does not claim that full RFS tracking is necessary. Instead, it uses the most useful idea from RFS:

> uncertain observations can be represented and processed as a finite set.

## Closing

In summary, this project studies primitive shape recognition using handheld GelSight contacts.

The main problem is that each tactile image is local and uncertain. One touch may not identify the object.

The proposed solution is to:

1. classify each touch into local geometric tactile features,
2. preserve uncertainty using probability distributions,
3. accumulate evidence over an unordered set of contacts,
4. use feature coverage to reward complementary observations,
5. stop once the object belief becomes confident and stable.

The research hypothesis is:

> Primitive object shape can be recognized efficiently by accumulating uncertain local tactile semantics from unregistered handheld contacts.

This makes the project a strong software and algorithmic contribution for handheld visuo-tactile object recognition.
