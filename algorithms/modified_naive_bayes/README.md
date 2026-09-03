# Modified Naive Bayes

Modified Naive Bayes keeps the interpretable posterior update of Naive Bayes
but adapts it to sparse tactile contacts.

It adds:

- repeated-touch damping, so duplicate local features contribute less evidence;
- feature weights, so distinctive tactile features carry more influence;
- coverage evidence, so different expected features strengthen confidence;
- an unexpected-feature penalty, so incompatible contacts reduce shape belief.

Configure it with:

```powershell
python -m algorithms.modified_naive_bayes.optimization --model models\pth\features.pth
```

Optimize touch plans for the configured Modified Naive Bayes model with:

```powershell
python tools\optimize_modified_nb_touch_plan.py --max-touches 30 --threshold 0.90 --timestamp
```

The touch-plan optimizer exports a summary CSV, a per-touch CSV, a JSON report,
and a confidence chart. It minimizes the touch count needed to reach the target
confidence threshold, then maximizes target-shape confidence and posterior
margin.
