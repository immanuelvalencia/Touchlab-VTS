# Shape Algorithm Registry

Each shape-classification algorithm lives in its own folder. The application
does not hardcode algorithm imports or comparison controls; it reads
`algorithms/registry.py`.

## Existing Layout

```text
algorithms/
  registry.py
  bayesian/
    __init__.py
    algorithm.py
    config.json
  naive_bayes/
    algorithm.py
    config.py
    config.json
    factory.py
    optimization.py
  modified_naive_bayes/
    algorithm.py
    config.py
    config.json
    factory.py
    optimization.py
  single_touch_baseline/
    algorithm.py
    config.py
    config.json
    factory.py
    optimization.py
  rule_based/
    algorithm.py
    config.py
    config.json
    factory.py
    optimization.py
  bag_of_features/
    algorithm.py
    config.py
    config.json
    factory.py
    optimization.py
  dirichlet_multinomial/
    algorithm.py
    config.py
    config.json
    factory.py
    optimization.py
  rfs/
    __init__.py
    algorithm.py
    config.py
    config.json
    factory.py
```

`naive_bayes` is the baseline probabilistic accumulator over local-feature
likelihoods, `modified_naive_bayes` adds tactile-specific repeated-touch
damping, feature weights, coverage evidence, and unexpected-feature penalties,
`single_touch_baseline` maps the latest feature directly to a shape belief,
`rule_based` applies transparent positive and negative feature rules,
`bag_of_features` performs histogram-prototype classification,
`dirichlet_multinomial` models unordered feature counts, and `rfs` contains the
RFS-inspired set-evidence implementation. The latter's UI name remains
`Set Evidence` because it is not a full multi-target RFS filter.

## Adding an Algorithm

1. Create a folder such as `algorithms/my_algorithm/`.
2. Add `__init__.py` and `algorithm.py`.
3. Implement `reset()` if the algorithm keeps state.
4. Implement an update method that accepts either one hard feature label or a
   complete feature-probability dictionary, according to the registration.
5. Add one registration call to the bottom of `algorithms/registry.py`.

Example registration:

```python
register_algorithm(
    key="my_algorithm",
    display_name="My Algorithm",
    module_path="algorithms.my_algorithm.algorithm",
    class_name="MyAlgorithm",
    update_method="add_touch",
    input_type="probabilities",
    description="Short description shown to developers.",
)
```

Valid input types are:

```text
hard_label     # receives the highest-probability local feature as a string
probabilities  # receives the complete {feature_name: probability} dictionary
```

The update method should return either:

- A normalized `{shape_name: probability}` dictionary.
- A result object with `prediction` and `belief` attributes. It may also expose
  `should_stop`, `is_uncertain`, `entropy`, `touch_count`,
  `feature_coverage`, and `stopping_reason` for the detailed comparison view.

After registration, `predict.py` automatically includes the algorithm in the
shape selector and comparison controls.

## Model-Specific Configuration

Algorithms with a registered `model_config_suffix` can load a same-stem sidecar
beside the selected model weights:

```text
my_model.pth
my_model.txt
my_model.nb.json
my_model.mnb.json
my_model.single_touch.json
my_model.rules.json
my_model.rfs.json
my_model.bof.json
my_model.dm.json
```

The suffixes select Naive Bayes, Modified Naive Bayes, Single Touch Baseline,
Rule-Based, Set Evidence, Bag of Features, and Dirichlet-Multinomial
configurations respectively. If a sidecar does not exist, the algorithm uses
the `config.json` in its own folder. Naive Bayes and Modified Naive Bayes use
the encoder's full feature-probability vector when the prediction app provides
one; the other hard-label algorithms use the top predicted feature.

## Optimization

Run the unified optimizer to fit, evaluate, compare, and export configurations:

```powershell
python optimize.py
```

For headless runs, pass a model and comma-separated algorithm keys:

```powershell
python optimize.py --run --model models\pth\features.pth --algorithms naive_bayes,modified_naive_bayes,single_touch_baseline,rule_based,bag_of_features
```

Each algorithm also exposes a standalone optimizer module:

```powershell
python -m algorithms.single_touch_baseline.optimization --model models\pth\features.pth
python -m algorithms.naive_bayes.optimization --model models\pth\features.pth
python -m algorithms.modified_naive_bayes.optimization --model models\pth\features.pth
python -m algorithms.rule_based.optimization --model models\pth\features.pth
python -m algorithms.bag_of_features.optimization --model models\pth\features.pth
```
