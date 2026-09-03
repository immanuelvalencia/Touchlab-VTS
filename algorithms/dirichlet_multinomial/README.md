# Dirichlet-Multinomial Classifier

This classifier models the complete unordered count vector of deterministic
local-feature labels. Each shape has a positive Dirichlet concentration vector.
The Dirichlet-Multinomial marginal probability discounts repeated observations
relative to an independent multinomial or Naive Bayesian update.

The multinomial coefficient is omitted at runtime because it is identical for
all candidate shapes and cancels during posterior normalization. A model-specific
configuration can be placed beside a model as `<model>.dm.json`.

For feature-count vector \(\mathbf n_T\), the class model is:

$$
P(\mathbf n_T\mid c)=
\frac{T!}{\prod_f n_T(f)!}
\frac{\Gamma(\alpha_{c,0})}{\Gamma(T+\alpha_{c,0})}
\prod_f
\frac{\Gamma(n_T(f)+\alpha_{c,f})}{\Gamma(\alpha_{c,f})}.
$$

Fit and validate the algorithm with:

```powershell
python optimize.py
```

Select Dirichlet-Multinomial in the application. Runs are saved under
`optimize/<model-name>/run_<number>/`; optional sidecar installation creates
`<model>.dm.json`.

Useful options:

```text
--initial-concentration 10
--max-iterations 500
--permutations 5
--test-permutations 20
--seed 42
```
