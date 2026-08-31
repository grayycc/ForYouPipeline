# Task: within-user ranking on KuaiRand-Pure

You are an autonomous ML research agent. Each iteration you write one complete, standalone
Python program that trains a model and produces ranked scores. You are judged on the
validation score you reach, and on the quality of the reasoning behind each attempt.

## The problem

For each user, rank the videos that user was actually shown in the evaluation window.
This is **not** retrieval over a catalogue — the candidate set is fixed and small
(the evaluation split averages about **5 impressions per user**). You are only ever judged
on the ordering *inside* one user's own list.

**Label:** `long_view` (native 0/1 column). It is logged on **every** impression, not only
clicked ones.

## Metrics — `evaluate.py` is the sole authority and MUST NOT be modified

`primary = mean(GAUC, nDCG@5)` — this single number decides everything.

- **GAUC**: per-user AUC, averaged weighted by each user's positive count. Users who are
  all-positive or all-negative are **excluded** (no contrast to measure).
- **nDCG@5**: gain is `2^rel - 1`; users with zero positives score 0.0 and **are included**
  in the average.
- Call it as `evaluate(user_ids, labels, scores)` -> `{'GAUC':…, 'nDCG@5':…, 'primary':…}`.

## Reference numbers

| | valid primary | test primary |
|---|---|---|
| random scoring (broken-harness check) | ~0.4827 | ~0.4753 |
| item popularity (trivial) | — | 0.5715 |
| **FM baseline — the bar you must clear** | **0.6016** | **0.5946** |
| oracle (true labels as scores) | — | **0.8645** |

The oracle ceiling is 0.8645, **not 1.0**: 27.1% of test users have no positive label
(nDCG 0, unfixable by any model) and 9.2% are all-positive. Real remaining headroom above
the baseline is ~0.27. Calibrate your expectations against 0.8645.

**Noise floor: the baseline's std across seeds is 0.0008.** Any apparent gain below ~0.002
is noise, not a result. Do not build on top of a change you have not separated from noise.

## Budget

- **50 iterations maximum.** Each wasted iteration is 2% of the entire run.
- Converged when validation primary has not improved by more than 0.002 over 3 consecutive
  iterations.
- Wall-clock ceiling 6 hours. The FM baseline trains in ~40 seconds on one CPU core, so
  compute is *not* your scarce resource — iterations are.

## Hard constraints

1. **No external training data.** KuaiRand files only. No pretrained weights.
2. **The leakage rule — memorise this.** For any row at time *t*, every feature must be
   computable using only information from strictly before *t*. Violating it makes validation
   look excellent and destroys the real score. Global (whole-period) target statistics are
   the classic way to lose here.
3. `evaluate.py`, `submit.py`, `data.py`, `baseline.py` are read-only. Do not edit or
   monkey-patch them. Write your own self-contained solution; import from them freely.
4. **Every iteration must produce a valid submission** — it is checked with
   `submit.py --check` and a node whose submission fails is discarded regardless of its score.
5. **One atomic change per iteration.** Change exactly one thing so its effect is
   attributable. Multi-change iterations make attribution impossible.

## Data contract

`from data import load, encode, FIELDS, SPLITS, LABEL`

- `load(data_dir)` -> `{'train': rows, 'valid': rows, 'test': rows}`, already date-sliced.
  Each row is a **tuple**, positionally:

  | idx | field | type |
  |---|---|---|
  | 0 | `date` | int, e.g. `20220408` |
  | 1 | `user_id` | str |
  | 2 | `video_id` | str |
  | 3 | `author_id` | str (`'UNK'` if unknown) |
  | 4 | `tab` | str |
  | 5 | `duration_ms` | float |
  | 6 | `long_view` label | int 0/1 |

- `encode(splits)` -> `(enc, dim)` where `enc[split] = (X, y, users)`;
  `X` is int32 `(N, len(FIELDS))` of **already-offset** feature indices sharing one embedding
  table of size `dim`; `y` is float32; `users` is a list of user_id strings.
  `FIELDS = ['user_id','video_id','author_id','tab','dur_bucket']`. Unseen values map to a
  per-field UNK slot. You may write your own encoder instead — `encode` is a convenience,
  not a requirement.

- Splits are **date-based**: train `20220408–0421` (1,141,112 rows) /
  valid `20220422–0428` (124,909) / test `20220429–0508` (170,588).
  Train has ~26,210 users at ~44 rows/user; the evaluation splits have ~5 rows/user.
  Behaviour and popularity drift across these weeks.

### Raw files in `data_dir` (beyond what `load()` reads)

| file | notes |
|---|---|
| `log_standard_4_08_to_4_21_pure.csv` | train window; 19 columns, **11 of them post-interaction — see below** |
| `log_standard_4_22_to_5_08_pure.csv` | valid+test window |

Their 19 raw columns, exactly, are: `user_id`, `video_id`, `date`, `hourmin`, `time_ms`,
`is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`, `long_view`,
`play_time_ms`, `duration_ms`, `profile_stay_time`, `comment_stay_time`, `is_profile_enter`,
`is_rand`, `tab`. **`author_id` is not one of them** — these logs only carry `video_id`;
`author_id` exists only after joining `video_features_basic_pure.csv` on it, which is exactly
what `load()` already does for you. Reading these files directly with `csv.DictReader` and
indexing `row['author_id']` raises `KeyError`; this has happened in past runs. Call `load()`
unless you specifically need a raw column `load()` does not carry, such as `is_click` or
`hourmin`.

**Eleven of those 19 columns describe what happened *during* the impression, and none of them
may be used as a feature for that row.** They are:

`play_time_ms`, `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`,
`profile_stay_time`, `comment_stay_time`, `is_profile_enter`, and `long_view` itself.

At serving time you are ranking a video the user has *not yet watched*, so none of these exist.
`play_time_ms` is the label in disguise: `long_view` is derived from it, and the single rule
`play_time_ms >= duration_ms` reproduces the label on **79.9%** of training rows by itself. A
run that used `play_time_ms / duration_ms` as a feature scored validation primary **0.8482**
against an oracle ceiling of 0.8484 and a GAUC of **0.9998** — it was reading the answer, and
that submission would have collapsed on the hidden test set. Any feature derived from these
columns for the row being scored is leakage, including ratios, differences and clipped versions
of them.

They are legitimate in exactly one direction: aggregated over a user's or a video's **strictly
past** rows to describe *history*, never the current row. `duration_ms`, `tab`, `date`,
`hourmin` and the IDs are known before the impression and are safe.
| `log_random_4_22_to_5_08_pure.csv` | **1,186,059 rows of randomly-exposed impressions** over the valid+test window. Videos here were shown at random rather than chosen by the production recommender, so metrics computed on it are not biased by the logging policy. |

**Do not hand-roll the random-exposure scoring — import it.** `kit/unbiased.py` does the
loading, the date filter, the `author_id` join and the encoding, and its equivalence to
`data.encode()` is asserted in `tests/test_unbiased.py`. Reimplementing it has produced
mismatched `dur_bucket` edges, all-`UNK` authors and a stubbed-out metric in past runs, and
none of it is a research question:

```python
from unbiased import load_random_valid, encode_like_train, unbiased_primary

rand_rows = load_random_valid(data_dir)                       # 288,338 rows, valid window
X_rand, y_rand, u_rand, _ = encode_like_train(splits['train'], rand_rows)
unbiased = unbiased_primary(data_dir, splits['train'], lambda rows: model.predict(X_rand))
```

`encode_like_train(train_rows, target_rows)` returns `(X, y, users, dim)` using the vocabulary
and quantile edges derived from `train_rows` alone. If your model does not use
`data.encode()`-style features, pass your own scoring function to `unbiased_primary` instead —
it only needs one score per row.

Its label distribution differs sharply from the standard logs — about 63% of users there have
no positive at all, against 27% in validation — so its absolute primary is much lower. That is
expected. What makes it useful is the comparison between your own runs, not its level.
| `user_features_pure.csv` | 27,285 users; activity, follower/fan counts, 18 anonymised one-hot features |
| `video_features_basic_pure.csv` | 7,583 videos; author, type, upload date/type, duration, music, tag |
| `video_features_statistic_pure.csv` | 7,583 videos of aggregate counters (`play_cnt`, `long_time_play_cnt`, `like_cnt`, …). **These are whole-period aggregates with no time dimension — they almost certainly include the test window, so using them directly is label leakage.** |

### The `FM` class in `baseline.py`, if you import it

`from baseline import FM` — `FM(dim, k=16, lr=0.001, l2=1e-6, seed=0)` with exactly three methods:

| method | signature | returns |
|---|---|---|
| `logits` | `logits(X)` | `(z, E, S)` — `z` is the score per row, shape `(B,)` |
| `step` | `step(X, y)` — **one Adam update on one minibatch** | the batch's mean BCE loss (float) |
| `predict` | `predict(X, bs=200_000)` | scores for all rows, shape `(N,)` |

There is no `fit`, no `train_batch`, and no epoch loop inside the class — you write the epoch
loop yourself. Weights are the public attributes `V` (embeddings), `W` (first-order), `b`.

Log columns available beyond those `load()` exposes: `hourmin`, `time_ms`, `is_click`,
`is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`, `play_time_ms`,
`profile_stay_time`, `comment_stay_time`, `is_profile_enter`, `is_rand`.

### Two additional content files

| file | size | key column | content |
|---|---|---|---|
| `kuairand_video_categories.csv` | 3.7 GB | `final_video_id` | four levels of hierarchical topic id + name + confidence |
| `kuairand_video_captions.csv` | 3.2 GB | `final_video_id` | `caption` (Chinese text), `show_cover_text`, `duration` |

These ship with the KuaiRand release, so they are in scope. They are sized for the whole
KuaiRand family (32M rows), not for Pure. The following was measured on this machine, so you
do not have to rediscover it:

- **`final_video_id` 0–7582 is exactly Pure's `video_id` space**, and those are the *first*
  7,583 records of both files, in ascending order. (Verified by matching the caption file's
  `duration` against `video_features_basic_pure.csv`'s `video_duration`: 7,583 of 7,583 agree.)
  **Break out of the reader once the id exceeds 7582.** Reading either file to the end costs
  ~55 s and ~4 GB of resident memory to collect ids belonging to the 1k/27k variants that are
  never scored here; the early exit costs ~0.02 s and ~2 MB. Both files together will approach
  the executor's memory cap if read in full.
- **Join key normalisation.** The id column reads as `'0'`, `'1'`, …, but category *values*
  carry a `.0` suffix (`'39.0'`). Key your map with `str(int(float(row['final_video_id'])))`
  so it matches the plain string `video_id` in `load()`'s rows. A silent key mismatch here
  produces an all-UNK feature that looks exactly like "the idea did not work".
- **Category coverage over Pure's 7,583 videos**, after filtering `-124.0` / `UNKNOWN` / `nan`:

  | level | distinct values | coverage |
  |---|---|---|
  | first | 38 | 99.8% |
  | second | 155 | 85.6% |
  | third | 245 | 37.6% |
  | fourth | 77 | 7.0% |

- **`video_features_basic_pure.csv` already carries a `tag` column locally** (626 KB, no large
  file needed). It is a *multi-valued* list (e.g. `'20,43'`) and agrees with
  `first_level_category_id` on 4,278 of 7,583 videos — a related but distinct taxonomy, not a
  substitute for it. It was **not** among the 13 fields in the organisers' negative feature
  ablation.
- **Captions are Chinese and unsegmented.** A `[A-Za-z0-9一-龥]+` tokeniser splits on
  `#` and spaces, so hashtags come out as clean tokens but each free-text sentence becomes one
  long token. Measured: 24,088 distinct tokens across Pure's videos, of which only 250 appear
  in 10 or more videos and 33 in 50 or more; 1,985 are single-occurrence long blobs. The most
  frequent tokens are platform campaign tags (`快手热点`, `集结吧光合创作者`) and account
  handles (`o40300129`) rather than topics.

The leakage rule applies to these exactly as it does to the logs: **never use whole-period or
post-time statistics directly as features.** Anything you aggregate must come from information
available strictly before the row being scored.

## What the FM baseline currently does

Read `baseline.py` for the exact code. In summary: a factorization machine over the 5
categorical fields above, embedding dim k=16, Adam at lr=0.001, batch size 8192, trained
with **pointwise binary cross-entropy** on independently-shuffled rows, early-stopping on
validation primary with patience 4.

## Findings already established — do not spend iterations rediscovering these

These were measured by the organisers and by prior runs. Treat them as known.

- Using all 13 static feature fields instead of 5: **0.5940 vs 0.5950** — no gain, if
  anything slightly worse.
- Embedding dim k = 8 / 16 / 32: **0.5895 / 0.5902 / 0.5887** — flat. Model capacity is
  *proven* not to be the bottleneck; 1.14M rows will not support a much bigger model.
- **What prior runs found that *did* work, with the measured size.** These are results from
  earlier runs of this same agent on this same split, recorded so a run does not have to spend
  its budget rediscovering them. They are measurements, not instructions — you decide whether
  to build on them, combine them, or go elsewhere.
  - **Seed ensembling with within-user rank averaging: +0.0016.** Train 5 copies of the model on
    seeds 0-4, convert each model's scores to within-user fractional ranks, then average the
    ranks rather than the raw scores. Averaging raw scores gave about half as much; the
    rank-space step is where most of it came from, because the metric only reads ordering.

    **This has a real cost you should size against your own iteration budget.** A promising
    node's whole script is re-run twice more to confirm the score on other seeds, so an
    ensemble trained *inside* your script multiplies: a 5-model internal ensemble becomes up
    to 15 model trainings for that one node, and one measured instance of this took ~19
    minutes wall-clock for a model that otherwise trains in seconds. That is not wasted --
    it produced the gain above -- but it is 15-20x a non-ensembled node's cost, so an
    ensemble size chosen without this in mind can eat a large share of the wall-clock ceiling
    for a handful of iterations.
  - **Bayesian-smoothed per-video and per-author long-view rate, bucketed: +0.0008.** Computed
    from training rows only and smoothed toward the global rate so rare IDs are not trusted.
  - Those two together account for +0.0024 of a best-ever +0.0025, and they compose: the
    ensembling result was measured on top of the smoothed-CTR model, not instead of it.
- **A feature that is constant within a user contributes exactly zero.** Only the ordering
  inside each user's list is scored, so any term identical across all of one user's rows
  cancels out entirely. This was measured: item-popularity alone and item-popularity crossed
  with a user bias term scored identically to the last digit. User-side information can
  therefore only matter through terms that vary across the videos shown to that same user.
- **A static attribute of `video_id` is very nearly a no-op, because the model already has a
  free embedding per `video_id`.** `video_type`, `first_level_category_id`, and any
  train-window CTR bucket are all pure functions of `video_id` (measured: 7583 videos, zero
  video_id -> video_type conflicts). A function of an ID adds information beyond that ID's own
  embedding *only* where the ID was too rare to learn an embedding for — i.e. cold-start. In
  the validation window that is **17 rows out of 124,909, or 0.01%**. So the ceiling on any
  such feature is 0.01% of rows, which is far below the 0.002 noise floor. This is measured,
  not predicted: it is why every item-side categorical tried so far (video CTR bucket, author
  CTR bucket, category, video_type, finer CTR bins) landed between +0.0002 and +0.0005.
  Their real value is as a *smoothed prior for rare IDs*, not as new signal.
- **The coverage asymmetry.** The same measurement run on the user side: **98.4%** of
  validation rows belong to a user who has training history, median 35 prior interactions,
  p10 = 6. Per-user aggregates still cancel (see above), so what this coverage buys is
  *interaction* terms — a statistic of this user crossed with an attribute of the candidate
  video, which does vary across that user's list. Attempts at this have so far failed by
  overfitting rather than by absence of signal (one hit train 0.9742 / valid 0.5808, a gap of
  0.39), which is a statement about the estimator, not about the mechanism.

## Your output contract — every iteration

Write **one standalone Python file** that:

1. Accepts exactly these arguments:
   `--data_dir <path>` `--out_dir <path>` `--seed <int>`
   `--train_split {train,train+valid}` (optional, default `train`)

   `--train_split train+valid` is used **once, after the run has converged**, and its only job
   is to regenerate the final test submission. The test window directly follows validation, so
   refitting the winning configuration on both splits gives it a week of more recent data.
   When it is passed:
   - train on `train` and `valid` rows combined;
   - do **not** early-stop on validation — it is inside the training set now, so the stopping
     signal is gone. Use the epoch count the `train` run settled on, or a fixed schedule;
   - write **only** `<out_dir>/submission_test.csv`. Do **not** score validation, do not write
     `submission_valid.csv`, and do not print the metric lines. Nothing reads them in this mode.

   Keep this branch narrow: it must not change how the default `--train_split train` path
   computes anything. Scoring validation with statistics fitted on train+valid would be
   leakage, which is exactly why this mode does not score validation at all.

   Ignoring the flag is not an error, but a solution that does not implement it simply forgoes
   the refit.
2. Trains on the train split and evaluates on the validation split using `evaluate()`.
3. Prints these lines to stdout, exactly, on their own lines (they are regex-parsed —
   no other format is read):

   ```
   TRAIN_PRIMARY=<float>
   VAL_GAUC=<float>
   VAL_NDCG5=<float>
   VAL_PRIMARY=<float>
   UNBIASED_PRIMARY=<float>
   ```

   All five are required.

   - `TRAIN_PRIMARY` is the same `evaluate()` call applied to the **training** split. The
     train-versus-validation gap is how overfitting to the training data becomes visible; a
     model can improve validation while that gap widens, and you should know when it does.
   - `UNBIASED_PRIMARY` is the primary score on `log_random_4_22_to_5_08_pure.csv`, the
     random-exposure log, restricted to rows in the validation date window. Because those
     impressions were shown at random rather than chosen by the production recommender,
     metrics on them are not biased by the logging policy. It is used as an acceptance gate:
     a change that raises normal validation while lowering the unbiased score is treated as
     overfitting to the logging policy's own biases and is rejected.
4. Writes both `<out_dir>/submission_valid.csv` and `<out_dir>/submission_test.csv` using
   `submit.write_submission(path, rows, scores)`, where `rows = load(data_dir)[split]` and
   `scores` is your model's score per row **in that exact row order**.

   Submission schema is `row_id,user_id,video_id,score` with `row_id` consecutive from 0.
   `row_id` is the key, **not** `(user_id, video_id)` — 3.06% of test pairs are duplicates,
   repeating up to 12 times.
5. Respects `--seed` for every source of randomness, so the same seed reproduces the same
   score and different seeds give an honest spread.
6. Imports only from the standard library and the packages listed below. Keep runtime under
   25 minutes.

### Available packages

`numpy`, `scipy`, `pandas`, `scikit-learn`, `lightgbm`, `xgboost`, `torch`.

Nothing else is installed, and the sandbox has no network, so an import outside this list will
fail. There is no GPU; `torch` runs on CPU.

> **Hard constraint, verified on this machine.** `torch` cannot be used in the same script as
> `lightgbm` or `xgboost`. Each bundles its own OpenMP runtime and the process dies on a
> segfault with no traceback the moment the second one does real work. Neither
> `KMP_DUPLICATE_LIB_OK=TRUE` nor `OMP_NUM_THREADS=1` avoids it. Pick one side per solution.
> `numpy`, `scipy`, `pandas` and `scikit-learn` are safe alongside either.

One property of the FM baseline worth knowing, since it constrains what a feature can even be:
an FM takes each feature as an index into a single shared embedding table, so a continuous
value has to be discretised into buckets before it can be used at all. Model families that
accept continuous inputs directly do not have that constraint.

## How to reason

Each iteration you propose **one** hypothesis. State what you are changing, the mechanism
by which you expect it to move GAUC or nDCG@5, and roughly how much you expect. Then, after
you see the result, say whether the mechanism held — and if it did not, whether it failed
because of a bug, because the idea was wrong, or because the difference was inside the noise
floor. Those three call for different responses.

The metric definitions, the within-user structure, the ~5-impressions-per-user shape of the
evaluation data, and the established findings above are the material you reason from.
Nobody is going to hand you a list of things to try — identifying what is worth trying, and
why, is the actual work.

