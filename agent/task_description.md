# Task: within-user ranking on KuaiRand-Pure

You are an autonomous ML research agent. Each iteration you write one complete, standalone
Python program that trains a model and produces ranked scores. You are judged on the
validation score you reach and on the quality of the reasoning behind each attempt.

## 1. The problem

For each user, rank the videos that user was actually shown in the evaluation window. This is
**not** retrieval over a catalogue — the candidate set is fixed and small, averaging about
**5 impressions per user**. You are only ever judged on the ordering *inside* one user's own
list.

**Label:** `long_view`, a native 0/1 column, logged on **every** impression rather than only
clicked ones.

Two consequences worth holding onto, because they invalidate techniques that work elsewhere:

- **Anything constant within a user contributes exactly zero.** A term identical across all of
  one user's rows cancels out of that user's ordering. User-side information can only matter
  through terms that *vary across the videos shown to that same user*.
- **Absolute calibration is worthless.** Only relative order within a user is read.

## 2. The metric — `evaluate.py` is the sole authority and MUST NOT be modified

`primary = mean(GAUC, nDCG@5)`. This single number decides everything.

| | definition |
|---|---|
| **GAUC** | `sum(npos_u * AUC_u) / sum(npos_u)` over users — each user's AUC weighted by **that user's own positive count**, which is *not* a plain average over users: a user with 30 positives counts thirty times one with a single positive. Users who are all-positive or all-negative are **excluded** — no contrast to measure. |
| **nDCG@5** | gain is `2^rel − 1`. Users with zero positives score 0.0 and **are included** in the average. |

Call it as `evaluate(user_ids, labels, scores)` → `{'GAUC': …, 'nDCG@5': …, 'primary': …}`.

## 3. Reference numbers

| | valid primary | test primary |
|---|---|---|
| random scoring (broken-harness check) | 0.4834 | 0.4753 |
| item popularity (trivial rung) | 0.5807 | 0.5715 |
| **FM baseline — the bar you must clear** | **0.6016** | **0.5946** |
| oracle (true labels used as scores) | 0.8484 | 0.8645 |

The oracle is **not 1.0** because 27.1% of test users have no positive label at all — their
nDCG is 0 for any model — and 9.2% are all-positive and excluded from GAUC. Only 63.7% of
users are rankable. Calibrate against 0.8645, not against 1.0.

**Noise floor: the baseline's spread across seeds is 0.0008.** A gain below ~0.002 is not
distinguishable from noise. Never build on a change you have not separated from it.

A score *below the item-popularity rung* means your code is broken, not that your idea was
wrong — a trained model cannot legitimately rank worse than counting impressions.

## 4. Budget

- **50 iterations maximum.** Each wasted iteration is 2% of the whole run.
- Converged when validation primary has not improved by more than **0.002 over 3 consecutive
  iterations** — this normally triggers before the iteration cap.
- Wall-clock ceiling 6 hours, but the FM baseline trains in ~40 seconds on one CPU core.
  **Compute is not your scarce resource. Iterations are.**

## 5. Hard constraints

1. **No external training data.** KuaiRand files only. No weights pretrained on this test set.
2. **The leakage rule.** For any row at time *t*, every feature must be computable using only
   information from strictly before *t*. Breaking it makes validation look excellent and
   destroys the real score. Whole-period target statistics are the classic way to lose here.
3. `evaluate.py`, `submit.py`, `data.py`, `baseline.py` are **read-only**. Do not edit or
   monkey-patch them. Write a self-contained solution; import from them freely.
4. **Every iteration must produce a valid submission.** It is checked with `submit.py --check`,
   and a node whose submission fails is discarded regardless of its score.
5. **One atomic change per iteration.** Change exactly one thing, so its effect is
   attributable. Multi-change iterations make attribution impossible.

## 6. Data contract

```python
from data import load, encode, FIELDS, SPLITS, LABEL
```

### `load(data_dir)`

Returns `{'train': rows, 'valid': rows, 'test': rows}`, already date-sliced. Each row is a
**tuple**, positionally:

| idx | field | type |
|---|---|---|
| 0 | `date` | int, e.g. `20220408` |
| 1 | `user_id` | str |
| 2 | `video_id` | str |
| 3 | `author_id` | str, `'UNK'` if unknown |
| 4 | `tab` | str |
| 5 | `duration_ms` | float |
| 6 | `long_view` | int 0/1 — the label |

### `encode(splits)`

Returns `(enc, dim)` where `enc[split] = (X, y, users)`. `X` is int32 `(N, 5)` of
**already-offset** feature indices sharing one embedding table of size `dim`; `y` is float32;
`users` is a list of user_id strings. `FIELDS = ['user_id','video_id','author_id','tab','dur_bucket']`.
Unseen values map to a per-field UNK slot.

`dur_bucket` edges come from `np.quantile` over the **training** split's `duration_ms` — they
are quantile-based, not round numbers. Any row you encode yourself must reuse those same edges
or it lands in different buckets from the rows the model trained on.

You may write your own encoder instead; `encode` is a convenience, not a requirement.

### Splits

Date-based, and behaviour drifts across the weeks:

| split | dates | rows | rows/user |
|---|---|---|---|
| train | 20220408–0421 | 1,141,112 | ~44 (26,210 users) |
| valid | 20220422–0428 | 124,909 | ~5 |
| test | 20220429–0508 | 170,588 | ~5 |

### Raw files in `data_dir`

| file | contents |
|---|---|
| `log_standard_4_08_to_4_21_pure.csv` | train window, 19 columns |
| `log_standard_4_22_to_5_08_pure.csv` | valid + test window, 19 columns |
| `log_random_4_22_to_5_08_pure.csv` | 1,186,059 randomly-exposed impressions over the valid+test window; **288,338 of them fall in the validation window** |
| `user_features_pure.csv` | 27,285 users: activity, follower/fan counts, 18 anonymised one-hot features |
| `video_features_basic_pure.csv` | 7,583 videos: author, type, upload date/type, duration, music, tag |
| `video_features_statistic_pure.csv` | 7,583 videos of aggregate counters. **Whole-period, no time dimension — they include the test window, so using them is label leakage.** |

Log columns beyond what `load()` exposes: `hourmin`, `time_ms`, `is_click`, `is_like`,
`is_follow`, `is_comment`, `is_forward`, `is_hate`, `play_time_ms`, `profile_stay_time`,
`comment_stay_time`, `is_profile_enter`, `is_rand`. The behavioural ones are *outcomes* of an
impression, so they cannot be features at prediction time — but nothing stops them being used
as training targets.

This release ships **no** caption or category files. Do not try to load them.

### The random-exposure log

Videos there were shown at random rather than chosen by the production recommender, so metrics
on it are not distorted by the logging policy. Two things must match how `load()`/`encode()`
treat the standard logs, or the number is not comparable:

- **No log file has an `author_id` column.** `load()` joins it from
  `video_features_basic_pure.csv` (`video_id → author_id`, `'UNK'` when missing). Do the same
  join rather than defaulting every row to `'UNK'`.
- **Reuse the training quantile `dur_bucket` edges**, per the note above.

Its label distribution differs sharply — about 63% of its users have no positive at all,
against 27% in validation — so its absolute score is far lower. That is expected. What makes
it useful is comparing your own runs against each other, not its level.

## 7. Tools available

### Libraries

Installed and importable: **numpy, pandas, scipy, scikit-learn, lightgbm, torch**, plus the
standard library. Nothing else — an import outside this list raises `ModuleNotFoundError` and
costs the whole iteration, so check the list rather than assuming.

Nothing obliges you to use `FM` or numpy. You may skip `baseline.py` entirely and hand
`evaluate()` scores from any model. **The FM is the number to beat, not the architecture to
keep.**

### The `FM` class, if you import it

`from baseline import FM` → `FM(dim, k=16, lr=0.001, l2=1e-6, seed=0)`, with exactly three
methods:

| method | signature | returns |
|---|---|---|
| `logits` | `logits(X)` | `(z, E, S)` — `z` is the score per row, shape `(B,)` |
| `step` | `step(X, y)` | one Adam update on one minibatch; returns mean BCE loss |
| `predict` | `predict(X, bs=200_000)` | scores for all rows, shape `(N,)` |

There is no `fit`, no `train_batch` and no epoch loop inside the class — you write the epoch
loop yourself.

**Public attributes — all of these are yours to use.** Shapes matter: `b` is a **0-d scalar**,
not an array, so `b[:] = …` raises `TypeError`. Assign it with `model.b = value`.

| | | shape |
|---|---|---|
| parameters | `V` (embeddings) | `(dim, k)` |
| | `W` (first-order) | `(dim,)` |
| | `b` (global bias) | scalar `np.float32` |
| Adam state | `mV`, `vV` | like `V` |
| | `mW`, `vW` | like `W` |
| | `t` (step counter) | int |
| settings | `lr`, `l2` | float |

`b` is a single number added to every row, so it is constant within a user and contributes
exactly nothing to the metric. Carrying it through a checkpoint is wasted work.

`step()` computes the BCE gradient **and** applies the Adam update in a single call, so a
different loss function cannot reuse it. That does **not** mean Adam is unavailable to you:
compute your own gradient and apply the same Adam update to those buffers directly.
`baseline.py` lines 63–69 are the exact six lines to mirror. Falling back to plain SGD at the
same learning rate converges far more slowly and will make a sound idea look like a failure.

## 8. Findings already established — do not spend iterations rediscovering these

Measured by the organisers. Treat as known.

- **All 13 static feature fields instead of 5:** 0.5940 vs 0.5950 — no gain, if anything
  slightly worse.
- **Embedding dim k = 8 / 16 / 32:** 0.5895 / 0.5902 / 0.5887 — flat. Capacity is *proven* not
  to be the bottleneck; 1.14M rows will not support a much larger model.
- **Constant-within-user terms score zero**, measured: item-popularity alone, and
  item-popularity crossed with a user bias, scored identically to the last digit.

## 9. Your output contract — every iteration

Write **one standalone Python file** that:

1. Accepts exactly `--data_dir <path>` `--out_dir <path>` `--seed <int>`.
2. Trains on train, evaluates on valid with `evaluate()`.
3. Prints these five lines to stdout, exactly, each on its own line. They are regex-parsed;
   no other format is read, and all five are required:

   ```
   TRAIN_PRIMARY=<float>
   VAL_GAUC=<float>
   VAL_NDCG5=<float>
   VAL_PRIMARY=<float>
   UNBIASED_PRIMARY=<float>
   ```

   - **`TRAIN_PRIMARY`** — the same `evaluate()` call on the **training** split. The
     train-versus-validation gap is how overfitting becomes visible: a model can improve
     validation while that gap widens, and you should know when it does.
   - **`UNBIASED_PRIMARY`** — primary on the random-exposure log, restricted to the validation
     date window. It is an acceptance gate: a change that raises normal validation while
     lowering this is treated as fitting the logging policy's biases, and is rejected. Compute
     it for real — emitting a placeholder disables the gate silently.

4. Writes both `<out_dir>/submission_valid.csv` and `<out_dir>/submission_test.csv` via
   `submit.write_submission(path, rows, scores)`, where `rows = load(data_dir)[split]` and
   `scores` is your score per row **in that exact row order**.

   Schema is `row_id,user_id,video_id,score`, `row_id` consecutive from 0. **`row_id` is the
   key, not `(user_id, video_id)`** — 3.06% of test pairs are duplicates, repeating up to 12
   times.

5. Respects `--seed` for every source of randomness, so one seed reproduces and different seeds
   give an honest spread.
6. Runs well under 15 minutes.

## 10. How to reason

Each iteration you propose **one** hypothesis. State what you are changing, the mechanism by
which you expect it to move GAUC or nDCG@5, and roughly how much.

Then, when you see the result, diagnose *which* of these happened — they demand different
responses, and confusing them is the most expensive mistake available to you:

| what happened | how to tell | what to do |
|---|---|---|
| the idea is wrong | ran correctly, moved the metric the wrong way by a real margin | retire the mechanism |
| the implementation is wrong | scored below the popularity rung, or the training curve looks broken | retry the same idea, implemented differently |
| the training run, not the idea, produced the number | `TRAIN_PRIMARY` pulled away from `VAL_PRIMARY` — the gap is much wider than the current best's | the mechanism is still untested; nothing has been learned about it yet |
| it was noise | moved less than ~0.002 | inconclusive; neither confirmed nor retired |
| it crashed | traceback | fix and rerun; nothing was learned about the idea |

The third row is the one that hides. It scores like a real negative — well above the
popularity rung, no traceback, a plausible number — so it reads as evidence against the idea
when it is evidence about how the model was fitted. That is why `TRAIN_PRIMARY` is in the
output contract: the gap is reported on every node precisely so you can separate the two, and
two nodes whose gaps differ sharply are not a fair comparison of their hypotheses. Watch the
gap across accepted nodes, not just within one.

Retiring a whole mechanism on one faulty implementation is unrecoverable — the run never
revisits it. Prefer one more experiment over a premature conclusion.

The metric definitions, the within-user structure, the ~5-impressions-per-user shape of the
evaluation data, and the established findings above are the material you reason from. Nobody
will hand you a list of things to try. **Identifying what is worth trying, and why, is the
actual work.**
