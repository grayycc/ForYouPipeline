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
| `log_standard_4_08_to_4_21_pure.csv` | train window; 19 columns |
| `log_standard_4_22_to_5_08_pure.csv` | valid+test window |
| `log_random_4_22_to_5_08_pure.csv` | **1,186,059 rows of randomly-exposed impressions** over the valid+test window. Videos here were shown at random rather than chosen by the production recommender, so metrics computed on it are not biased by the logging policy. |
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

Note: this KuaiRand-Pure release ships **no** caption or category files — do not try to load them.

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
- **A feature that is constant within a user contributes exactly zero.** Only the ordering
  inside each user's list is scored, so any term identical across all of one user's rows
  cancels out entirely. This was measured: item-popularity alone and item-popularity crossed
  with a user bias term scored identically to the last digit. User-side information can
  therefore only matter through terms that vary across the videos shown to that same user.

## Your output contract — every iteration

Write **one standalone Python file** that:

1. Accepts exactly these arguments:
   `--data_dir <path>` `--out_dir <path>` `--seed <int>`
2. Trains on the train split and evaluates on the validation split using `evaluate()`.
3. Prints these lines to stdout, exactly, on their own lines (they are regex-parsed —
   no other format is read):

   ```
   VAL_GAUC=<float>
   VAL_NDCG5=<float>
   VAL_PRIMARY=<float>
   ```

   Optionally, if you also evaluate on the random-exposure log:

   ```
   UNBIASED_PRIMARY=<float>
   ```

   When present, this is used as an additional acceptance gate: a change that raises normal
   validation while lowering the unbiased score is treated as overfitting to the logging
   policy's own biases and is rejected.
4. Writes both `<out_dir>/submission_valid.csv` and `<out_dir>/submission_test.csv` using
   `submit.write_submission(path, rows, scores)`, where `rows = load(data_dir)[split]` and
   `scores` is your model's score per row **in that exact row order**.

   Submission schema is `row_id,user_id,video_id,score` with `row_id` consecutive from 0.
   `row_id` is the key, **not** `(user_id, video_id)` — 3.06% of test pairs are duplicates,
   repeating up to 12 times.
5. Respects `--seed` for every source of randomness, so the same seed reproduces the same
   score and different seeds give an honest spread.
6. Uses only the Python standard library and numpy. Keep runtime well under 15 minutes.

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
