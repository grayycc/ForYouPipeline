# ForYouPipeline — an autonomous ML research agent for KuaiRand-Pure

An LLM-driven agent that designs, implements, reviews, and iterates on within-user ranking
solutions for the KuaiRand-Pure benchmark with no human in the loop during a run. Given only the
task definition and the starter kit, it reproduces the official FM baseline, then searches for
improvements — proposing a hypothesis, writing the training code, checking it for label leakage,
running it, diagnosing the result, and deciding what to try next — until the challenge's own
convergence rule says to stop.

## Result

|  | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official baseline (hidden test) | 0.6610 | 0.5282 | 0.5946 |
| **Our submission (hidden test)** | **0.6643** | **0.5306** | **0.5975** |
| Δ over baseline | +0.0033 | +0.0024 | **+0.0029** |

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official baseline (validation) | 0.6674 | 0.5357 | 0.6016 |
| **Our submission (validation)** | **0.6711** | **0.5373** | **0.6042** |
| Δ over baseline | +0.0037 | +0.0016 | **+0.0026** |

Validation is what the agent optimizes against; hidden-test is scored once, at the end, and
never seen during the run. Both tables are measured, not projected — see
[Reproducing the result](#reproducing-the-result). Full resource-usage and iteration numbers are
in [Results in detail](#results-in-detail) below.

Scored per the challenge formula — `delta(m) = score_agent(m) − score_baseline(m)`, then the
equal-weighted mean over both metrics — that is **+0.0029 on hidden test** (mean of +0.0033 GAUC
and +0.0024 nDCG@5) and **+0.0026 on validation**. Because `primary` is itself the mean of the
two metrics, the primary-row delta and the formula give the same number.

The hidden-test row is the submitted file, `runs/v10/best_submission_test.csv`. It scores
slightly higher than node 6's own test output (0.5975 vs. 0.5963) because after converging, the
agent refits the winning configuration on train **+** validation and regenerates only the test
submission — the test window directly follows validation, so that buys it a week of more recent
data. Validation is still scored by the train-only model, since validation is inside the refit's
training set and scoring it there would be meaningless.

**The agent never sees the hidden test set.** Verified in the submitted solution: `evaluate()`
is called only on train and valid, and the test split's label array is unpacked but used
exclusively for `len()` when sizing a scores buffer — never as a target, never for model
selection. Test rows are used only to build features from their own known-at-serving-time
columns and to write the submission.

## How it works

`run_agent.py` drives `agent/orchestrator.py` through an AIDE-style greedy tree search:
baseline → EDA → repeated `draft` / `improve` / `debug` nodes, until convergence.

- **`agent/roles/planner.py`** decides the next experiment — one atomic, falsifiable hypothesis
  per iteration, grounded in the metric definitions, the data's structure, and (optionally) a
  literature search via `agent/research.py`. It's also shown two kinds of accumulated evidence
  it didn't have to pay for itself: which mechanisms this run has already tried and how they
  scored, and — pooled across every prior run's logs — which mechanism *families* have actually
  paid off historically (`agent/diagnose.py:cross_run_yield`).
- **`agent/roles/coder.py`** writes the full standalone training script for that hypothesis.
- **`agent/roles/reviewer.py`** reads the diff against the parent solution for one specific
  defect — label leakage — before a node is ever allowed to become the submission. A flagged
  node still runs (a false positive shouldn't cost the experiment), but it can't win.
- **`agent/gates.py`** rejects a spec outright if it repeats a mechanism that's failed
  ≥3 times in this run with zero accepts and zero results inside the noise floor
  (`agent/diagnose.py:exhausted_mechanisms`) — the search can't get stuck re-litigating a
  settled question.
- **`agent/diagnose.py`** classifies every scored node deterministically —
  `under_trained` / `overfit` / `noise` / `regression` / `improvement` — so the planner reasons
  from a diagnosis, not a bare number. A model that never fit its own training data isn't
  evidence the *idea* was wrong.
- **The unbiased-exposure gate** scores every candidate against `log_random_4_22_to_5_08_pure.csv`
  (randomly-exposed impressions, unbiased by the logging policy) as well as normal validation,
  and rejects a change that improves validation while collapsing on random exposure — a proxy
  for overfitting to the production recommender's own biases rather than to user preference.

`agent/task_description.md` is the single source of truth the agent itself reasons from — data
contract, metric definitions, the output contract every generated solution must satisfy, and a
running record of what's been measured (including dead ends, so the agent doesn't re-spend a
turn rediscovering them).

## Setup

```bash
python3 -m venv .venv-1
source .venv-1/bin/activate
pip install -r requirements.txt
```

Download the data (see [Data](#data) below), then copy `.env.example` to `.env` and fill in an
AWS Bedrock bearer token, region, and one inference-profile name per agent role. **This is only
needed to run the agent itself** — re-scoring an existing submission needs nothing but numpy.

## Reproducing the result

**Exact — no LLM calls, deterministic:**

```bash
export PYTHONPATH="$PWD/kit"   # the solution imports data/baseline/evaluate/submit from kit/

# 1. Train on `train` only -> the validation-best checkpoint and its valid submission
python3 runs/v10/best_solution.py \
  --data_dir KuaiRand-Pure/data --out_dir /tmp/repro --seed 0
python3 kit/submit.py --score --split valid --data_dir KuaiRand-Pure/data /tmp/repro/submission_valid.csv

# 2. Refit the same configuration on train+valid -> the submitted test file
python3 runs/v10/best_solution.py --train_split train+valid \
  --data_dir KuaiRand-Pure/data --out_dir /tmp/repro_refit --seed 0
python3 kit/submit.py --check --split test --data_dir KuaiRand-Pure/data /tmp/repro_refit/submission_test.csv
```

Step 1 reproduces the validation numbers; step 2 reproduces
`runs/v10/best_submission_test.csv`, the file that is actually submitted. Step 2 deliberately
writes no validation submission and prints no metrics — validation is inside its training set
at that point, so any score it reported would be meaningless.

This is the code that produced the numbers above. It was re-verified this session from a clean
`git worktree` checkout with no uncommitted files present, at three independent ensemble-seed
offsets (0.6042 / 0.6044 / 0.6043 — reproduces to ±0.0002), and both submission files were
scored directly against the hidden-test labels.

**The search that found it** — genuinely stochastic, not guaranteed to rediscover this exact
result:

```bash
python3 run_agent.py --run_id my_run --max_iterations 50
```

`runs/v10` is one run among several from this codebase (`runs/v9`, `v11`–`v14` are also
committed) — they converged in the +0.0018 to +0.0026 range. The planner's literature search and
its own reasoning differ run to run by design; what's guaranteed to reproduce is the *code*, not
the discovery of it.

## Results in detail

From `runs/v10/summary.json` (the run's own record, not hand-computed):

| | |
|---|---|
| Iterations used | 8 of 50 cap |
| Converged at iteration | 7 (validation improved ≤0.002 over 3 consecutive scoring iterations — the literal rule, no floor: `min_iterations_before_convergence = 0`) |
| Best node | 6 |
| Total tokens (in + out) | 217,595 |
| Agent wall-clock | 4,549s ≈ 1.26h |
| GPU-hours | 0 (CPU only) |
| Manual interventions during the run | **0** |
| Buggy nodes (crashed, timed out, etc.) | 2 of 8 — both recovered from automatically |

The accepted trajectory — each step is a real, attributable improvement, not a single lucky
draw:

| node | mechanism | validation primary | Δ vs. baseline |
|---|---|---|---|
| 0 | FM baseline reproduction | 0.6015 | — |
| 2 | + 5-seed rank-averaged ensemble | 0.6028 | +0.0012 |
| 3 | + Bayesian-smoothed video/author CTR buckets | 0.6034 | +0.0018 |
| 6 | + BPR pairwise loss in place of pointwise BCE | **0.6042** | **+0.0026** |

Full per-iteration detail — hypothesis, the actual code diff against its parent, metrics, and
every error/recovery event including one node that crashed on a bug and one whose LLM call
itself failed — is in [`runs/v10/ITERATION_LOG.md`](runs/v10/ITERATION_LOG.md), generated by
`build_iteration_log.py` directly from the run's own files (real diffs, not descriptions).

## Limitations & what I'd improve with more time

- **In this specific run, every individual accepted step was inside the noise floor; only the
  chain clears it.** Node-to-node deltas are +0.0013, +0.0006, +0.0008 — none alone would be
  called a result on its own, and this run's win comes from compounding three of them. That's
  not a universal property of the task (other runs from this codebase have landed a single step
  above ε — e.g. +0.0023 in one case), but it happens often enough that the convergence rule
  (ε=0.002 over 3 consecutive scoring iterations) doesn't reliably distinguish "the search is
  actually done" from "the last few tries individually failed to clear noise." Several runs from
  this codebase converged at iteration 4–6 having found nothing at all, for exactly that reason.
- **KuaiRand-1K and KuaiRand-27K (bonus benchmarks) are not attempted.** The FM's per-minibatch
  update is `O(vocabulary size)` — cheap at Pure's ~40K-entry embedding table, but measured at
  ~19× slower per step at 1K's ~2.9M-entry table (313ms vs. 16.5ms, profiled directly against
  the real 1K data). A single unensembled training run is already close to the per-node timeout
  at that scale, and every mechanism that's actually produced a win here (ensembling,
  seed-confirmation) multiplies that cost 3–10×. Tractable with a sparse-update rewrite of the
  embedding gradient step (touch only the rows a minibatch actually references, not the whole
  table); not attempted this round. KuaiRand-27K (~1,000× Pure's vocabulary) would need that
  fix plus a streaming loader — the current `data.load()` materializes every row in memory, and
  27K's logs alone are on the order of tens of GB as plain Python tuples.
- **Multi-task auxiliary supervision was tried and refuted, not skipped.** `is_click` correlates
  extremely strongly with the `long_view` label (P(long_view|click)=0.72 vs. 0.003 without) and
  looked like the strongest untried lever in the dataset. Training a second head on it
  (`kit/aux_labels.py`) was implemented, code-reviewed as leakage-clean, and tried twice — both
  attempts regressed validation. Plausible reading in hindsight: a near-duplicate label adds
  little a shared-embedding model doesn't already have from the main label.
- **The reviewer's leakage judgments are LLM calls, not a formal proof.** Diff-scoped review
  (judging only what changed from an already-cleared parent) measurably fixed a real false-positive
  problem, but it remains a language model reading code, not a static analyzer — worth a second,
  independent pass before treating any single run's `CLEAN` verdict as certainty on a security-
  or leakage-sensitive line.

## Team contributions

Charlene, Grace, Shanyce, Beatriss, Priscilla.

The work was collaborative rather than split into owned components: each of us ran our own
exploration and research into the problem, and the design of the agent and its evaluation
harness came out of comparing those findings.

## Data

Download from [kuairand.com](https://kuairand.com) (direct Zenodo link, no registration needed):

```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## Files

| | |
|---|---|
| `run_agent.py` | Entry point for a full autonomous run |
| `build_iteration_log.py` | Renders a run's `log.jsonl` + on-disk node files into a per-iteration report with real code diffs (`python3 build_iteration_log.py <run_id>`) |
| `agent/orchestrator.py` | The search loop: node selection, execution, accept/reject, convergence |
| `agent/roles/` | One file per LLM-driven role — planner, coder, reviewer, debugger, EDA, baseline, draft |
| `agent/diagnose.py` | Deterministic (no LLM) classification of every scored node, and cross-run mechanism-yield tracking |
| `agent/gates.py` | Spec validation — duplicate/exhausted-mechanism rejection, protected-file checks, provenance |
| `agent/task_description.md` | The task, as the agent itself reads it |
| `kit/` | Starter kit: `data.py` / `evaluate.py` / `baseline.py` / `submit.py` are read-only by convention (never modified by the agent or by hand); `unbiased.py` and `aux_labels.py` are helpers added this project so generated solutions stop re-deriving the same alignment logic every iteration |
| `runs/<id>/` | One directory per run — `log.jsonl` (per-iteration hypothesis, metrics, diagnosis, error/recovery events), `summary.json` (resource usage, final result), `best_solution.py`, `best_submission_{valid,test}.csv` |
| `tests/` | Unit tests for the deterministic parts of the harness (gates, diagnosis, the alignment helpers) — not for the LLM-driven roles, which are validated empirically via runs |

---

## Starter-kit reference

Conventions below are fixed by the organizers and used exactly as given — `evaluate.py` is the
sole authority on scoring and is never modified.

| | |
|---|---|
| Task | **Within-user ranking** — each user only ranks their own impressions in the evaluation set; no full-corpus retrieval |
| Relevance label | `long_view` (native column, 0/1) |
| Metrics | `GAUC`, `nDCG@5`; **primary score = the average of the two** |
| Data split | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| Users with zero positives | nDCG counts as 0.0 and is included in the average; GAUC only counts users with `0 < #positives < #impressions`, weighted by positive count |
| nDCG gain | `2^rel − 1` (equivalent to identity under binary labels) |

### The real range of the metric: the ceiling for nDCG@5 is 0.729, not 1.0

Among the 23,875 users in the test set, 27.1% are all-negative (nDCG is always 0, no model can
fix this) and 9.2% are all-positive (nDCG is always 1) — 63.7% actually determine GAUC. So even
the oracle (true labels used as scores) only reaches:

| | random | FM baseline | **oracle ceiling** |
|---|---|---|---|
| GAUC | 0.4996 | 0.6610 | **1.0000** |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** |
| **primary** | 0.4753 | **0.5946** | **0.8645** |

The baseline already captures about a third of the usable range — measure progress against
0.8645 as the denominator, not 1.0. FM's std across 5 seeds is **0.0008**; the convergence rule
is **ε = 0.002 (≈2.5σ), N = 3 consecutive iterations**.

### Submission format

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

`row_id` is the primary key (consecutive from 0, matching `data.load()[split]`'s row order) —
`(user_id, video_id)` is not unique in the evaluation set (3.06% of test pairs repeat, up to 12
times).

```bash
python3 kit/submit.py --check --split test --data_dir KuaiRand-Pure/data submission.csv
python3 kit/submit.py --score --split valid --data_dir KuaiRand-Pure/data submission.csv
```
