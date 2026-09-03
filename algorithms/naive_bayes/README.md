# Naive Bayes

Naive Bayes is the baseline probabilistic accumulator for local tactile
features. It assumes each touch is conditionally independent given the primitive
shape and updates the shape posterior from the configured feature likelihoods.

The app passes the full local-feature probability vector when available. A
one-hot vector is equivalent to the standard hard-label Naive Bayes update.

Configure it with:

```powershell
python -m algorithms.naive_bayes.optimization --model models\pth\features.pth
```

