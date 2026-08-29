# KuaiRand-Pure Starter Kit

## Dependencies

Python 3.9+ and numpy. **Nothing else.** No torch, pandas, or sklearn required.

## Data

Download from https://kuairand.com (direct Zenodo link, no registration needed):

```bash
# Run inside the Starter Kit directory; extraction produces ./KuaiRand-Pure/
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Running

```bash
python3 baseline.py --model fm
```

`--data_dir` defaults to `./KuaiRand-Pure/data`; specify it explicitly if the data lives elsewhere.

`--model` accepts `fm` (official baseline) / `pop` (trivial baseline) / `random` (lower bound, for sanity-checking the evaluation code).
FM takes about 40 seconds end to end (CPU, single core).

## Task definition (the conventions are fixed — do not change them)

| | |
|---|---|
| Task | **Within-user ranking** — each user only ranks their own impressions in the evaluation set; no full-corpus retrieval |
| Relevance label | `long_view` (native column, 0/1) |
| Metrics | `GAUC`, `nDCG@5`; **primary score = the average of the two** |
| Data split | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Users with zero positives | nDCG counts as 0.0 and is included in the average; GAUC only counts users with `0 < #positives < #impressions`, weighted by positive count |
| nDCG gain | `2^rel − 1` (equivalent to identity under binary labels) |

See `evaluate.py` for the implementation; all conventions are documented in the file header comments.

## Baseline ladder

Scores on the test set. **The row to beat is FM.**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (lower bound, sanity check) | 0.4996 | 0.4511 | 0.4753 |
| item popularity (trivial) | 0.6308 | 0.5121 | 0.5715 |
| **FM (official baseline)** | **0.6610** | **0.5282** | **0.5946** |

### ⚠️ The real range of the metric: the ceiling for nDCG@5 is 0.729, not 1.0

Among the 23,875 users in the test set:

| | Share | Effect on the metric |
|---|---|---|
| All-negative users (none of the user's impressions are `long_view`) | **27.1%** | nDCG is always **0**; no model can fix this; excluded from GAUC |
| All-positive users | **9.2%** | nDCG is always **1**; excluded from GAUC |
| Users with discriminable labels | **63.7%** | The actual sample for GAUC |

So even using the true labels as prediction scores (oracle, perfect ranking) only gets you:

| | random | FM baseline | **oracle ceiling** | Range already captured by FM |
|---|---|---|---|---|
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

**Measure your progress against the oracle as the denominator.** Seeing 0.5946 and concluding "that's still far from a perfect 1.0" is a misreading — the baseline has already captured about a third of the usable range, and the remaining headroom is 0.27, not 0.41.

FM's std across 5 random seeds is **0.0008** for every metric. Based on that, the convergence criterion is **ε = 0.002 (≈2.5σ), N = 3**: if the validation primary score improves by no more than 0.002 for 3 consecutive iterations, treat it as converged.

> Sanity check: if running `--model random` through your evaluation code does not give primary ≈ 0.475 (±0.001), your harness is broken — fix that first.

## Submission format

CSV with a header, one row per row of the evaluation set:

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

| Field | Description |
|---|---|
| `row_id` | Consecutive, starting at 0, matching the row order of `data.load()[split]` (deterministic: read `log_standard_4_08_to_4_21_pure.csv` first, then `log_standard_4_22_to_5_08_pure.csv`, filter by date, and preserve the original file order) |
| `user_id` / `video_id` | Redundant fields, used only to verify alignment |
| `score` | Your model's score for that row; any real number, only relative order matters; NaN / Inf not allowed |

> **Why `row_id` is mandatory:** `(user_id, video_id)` is **not unique** in the evaluation set — 3.06% of the pairs in the test set are duplicates, repeating up to 12 times. So it cannot serve as a primary key.

Generating and validating:

```bash
python3 submit.py --make  --split test  submission.csv    # generate a sample submission using the official FM baseline
python3 submit.py --check --split test  submission.csv    # validate format and alignment
python3 submit.py --score --split valid submission.csv    # validate and score (available locally for valid)
```

`--check` will reject: wrong header, wrong row count, gaps in `row_id`, `user_id`/`video_id` misaligned with the evaluation set, and `score` values that are non-numeric or NaN/Inf. **Please run `--check` yourself before submitting.**

## Where to start making changes

The ordering below is **empirically tested**, not guesswork. Dead ends the organizers have already tried are marked explicitly — don't repeat them.

### Already tested: these two yield nothing, don't waste iterations on them

| What was tried | Result |
|---|---|
| **Adding static features** — wiring in all 13 of CWM's feature fields (+`music_id`/`video_type`/`upload_type` + 6 coarse user-side buckets) | primary **0.5940** vs **0.5950** with 5 fields — indistinguishable within noise, if anything slightly worse |
| **Adding model capacity** — embedding dimension k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887, essentially flat |

The reason: the `user_id × video_id` cross already absorbs most of the learnable signal. Coarse buckets like `follow_user_num_range` are redundant once you have `user_id`; and 1.14M rows can't support more capacity anyway. **The bottleneck is neither features nor capacity.**

⚠️ Also note: **first-order terms on purely user-side features contribute exactly 0 to the score.** Because ranking happens within a user, any term that is constant within a user does not change the intra-group ordering (measured: `item_pop × user bias` and plain `item_pop` give scores identical to the last digit). User-side features can only take effect through **cross terms with the item side**.

### Unexplored: the headroom should be here

Ordered by our estimate of how promising they are (**the organizers have not tested any of these — they're left for you**):

1. **Change the loss function.** Currently it's pointwise logloss, but the metrics (GAUC / nDCG) are **ranking metrics**. Switching to pairwise (BPR) or listwise (softmax over that user's impressions) aligns the objective with the evaluation convention — we think this is the most likely to work.
2. **User behavior sequences.** The existing features **make no use of behavior sequences at all**. Each KuaiRand user has hundreds to thousands of interactions in train; interest modeling in the DIN / SIM family is a completely blank direction here.
3. **Multi-objective.** The logs also contain `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, and `play_time_ms`, which can serve as auxiliary tasks for the main `long_view` task.
4. **Modeling watch time.** This is exactly the contribution of [CWM](https://github.com/hyz20/CWM): it treats watch time as **censored regression** (when a video plays to completion the true watch time is truncated, so it uses a one-sided loss rather than squared error). This is a direction with real research depth.
5. **Change the model.** DeepFM / DCN / xDeepFM. Since capacity has been measured not to be the bottleneck, **give this lower priority than 1-4.**
6. **Time features and distribution drift.** `hourmin`, `date`, and the drift between train and test.
7. **Unbiased validation (advanced).** `log_random_4_22_to_5_08_pure.csv` is a random-exposure log (1.18M rows) that can serve as an extra unbiased validation set to check whether your model only overfits biased traffic.

## Using your own model (including CWM)

`evaluate.py` is fully decoupled from the model; it only needs three equal-length arrays:

```python
from evaluate import evaluate
print(evaluate(user_ids, labels, scores))   # scores can come from any model
```

- `user_ids`: the user_id for each row of the evaluation set
- `labels`: that row's `long_view` (0/1)
- `scores`: your model's score for that row (any real number, only relative order matters)

So you can skip `baseline.py` entirely and use PyTorch, LightGBM, or [CWM](https://github.com/hyz20/CWM)'s xDeepFM instead — just hand your `scores` to `evaluate()` at the end. **`evaluate.py` is the sole authority on scoring.**

> A caveat on using CWM: it depends on `torch==1.6.0` (a 2020 release, which probably won't install on newer GPUs), and its loss optimizes counterfactual watch time while its evaluation label is a self-reconstructed `long_view2`. It's the research code for a watch-time debiasing paper — useful as an **advanced reference**, not recommended as a starting point.

## Files

| | |
|---|---|
| `evaluate.py` | Metric implementation + all scoring conventions. **Do not modify.** |
| `data.py` | Data loading, official splits, feature encoding. Add features here. |
| `baseline.py` | The three baselines. FM is the one to beat. |
| `baseline_scores.json` | Officially published scores + seed variance + convergence parameters. |
| `submit.py` | Generate / validate submission files. |
| `ablation_features.py` | Feature ablation experiments; reproduces the "adding features yields nothing" numbers. |
