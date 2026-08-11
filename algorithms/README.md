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
    touch_sequences.json
  rfs/
    __init__.py
    algorithm.py
    config.py
    config.json
    factory.py
```

The `rfs` folder contains the RFS-inspired set-evidence implementation. Its UI
name remains `Set Evidence` because it is not a full multi-target RFS filter.

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

## Model-Specific RFS Configuration

The default RFS configuration is `algorithms/rfs/config.json`. To use different
features, shapes, templates, priors, or parameters for a trained model, place a
sidecar configuration beside its weights:

```text
my_model.pth
my_model.txt
my_model.rfs.json
```

When `my_model.pth` is loaded, the registry automatically supplies
`my_model.rfs.json` to RFS. If that sidecar does not exist, RFS uses its default
configuration. The current RFS implementation uses only the encoder's top
feature label; model softmax confidence and encoder calibration are ignored.
