"""Agent-wide constants. Everything tunable lives here so no magic numbers hide in the loop."""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The official starter kit lives in kit/ and is never modified. Putting it on sys.path keeps
# `from data import load` working for the harness; executor.py does the same for the
# subprocesses that run generated solutions.
KIT_DIR = os.path.join(REPO_ROOT, 'kit')
if KIT_DIR not in sys.path:
    sys.path.insert(0, KIT_DIR)


def _load_dotenv(path=os.path.join(REPO_ROOT, '.env')):
    """Minimal .env loader so `python3 run_agent.py` works with no extra dependency.
    Real environment variables always win over the file."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, val = line.partition('=')
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and val and not os.environ.get(key):
                os.environ[key] = val


_load_dotenv()


def _env(name, default=None):
    """os.environ.get, but treats an empty value as unset (the .env template ships blanks)."""
    v = os.environ.get(name)
    return v if v else default


_custom_data_dir = _env('DATA_DIR')
if _custom_data_dir:
    DATA_DIR = os.path.abspath(os.path.expanduser(os.path.expandvars(_custom_data_dir)))
else:
    DATA_DIR = os.path.join(REPO_ROOT, 'KuaiRand-Pure', 'data')
    legacy_dir = os.path.join(REPO_ROOT, 'data', 'KuaiRand-Pure', 'data')
    if not os.path.isdir(DATA_DIR) and os.path.isdir(legacy_dir):
        DATA_DIR = legacy_dir

RUNS_DIR = os.path.join(REPO_ROOT, 'runs')

# Budget, from the problem statement.
MAX_ITERATIONS = 50               # hard cap per benchmark run
WALL_CLOCK_CEILING_S = 6 * 3600   # backstop

# Convergence. The README reports a seed std of 0.0008, so eps is about 2.5 sigma.
EPSILON = 0.002
CONVERGENCE_N = 3

# Convergence is a stopping rule for a search that has actually run, not a warm-up filter.
# Four scoring nodes exist by iteration ~4 (baseline plus three experiments), so with no floor
# the epsilon/N rule fires the moment three experiments in a row fail to beat a strong
# baseline -- which is the normal early state of any search, not evidence of a plateau.
# Measured: every prior run stopped at iteration 4 of 50. The rule below still decides when
# the run ends; this only stops it deciding before there is a search to stop.
#
# Raised from 15 after runs/v5 stopped at iteration 18 having spent 1.0h of a 6h ceiling and
# 19 of 50 iterations, shipping +0.0003. Stopping early is only cheap if the result is good:
# resource consumption is scored in three coarse tiers AND only among entries that beat the
# baseline, so converging at the noise floor forfeits the primary metric to save something that
# was not close to a tier boundary. 30 still leaves the cap and the wall-clock ceiling as the
# real limits, and the epsilon/N rule ends the run whenever it fires after that.
MIN_ITERATIONS_BEFORE_CONVERGENCE = 30

# Reference scores to measure against.
BASELINE_VALID_PRIMARY = 0.6016
BASELINE_TEST_PRIMARY = 0.5946
ORACLE_PRIMARY = 0.8645
SEED_STD = 0.0008

# Search policy.
MIN_DRAFTS = 3                    # distinct fresh solutions before improve-only

# Escaping a plateau. `improve` only ever mutates the incumbent, so once the drafts are spent
# the search has no way to make a large jump -- runs/v3 burnt its three drafts at iterations
# 2-4, when the agent knew least, and then spent six of its twelve remaining improves adding
# one more static item-side categorical to the same FM, every one inside the noise floor.
# A run of results that all land in the noise is the signal that incremental variation on this
# incumbent has stopped paying, and that a blank file is worth more than another edit.
STALL_NOISE_STREAK = 4            # consecutive `noise` verdicts that count as a plateau
STALL_DRAFT_COOLDOWN = 3          # iterations between forced drafts, so a plateau cannot
                                  #   spend the whole budget drafting
MAX_DRAFTS = 8                    # hard cap on drafts across the run
DEBUG_PROBABILITY = 0.5           # chance of debugging a buggy leaf when one exists
MAX_DEBUG_DEPTH = 2               # abandon a branch after 2 failed fixes (playbook: not 3)

# Seed averaging, only spent on candidates that already look promising.
CONFIRM_SEEDS = [0, 1, 2]
CONFIRM_TRIGGER = EPSILON / 2     # re-run with more seeds once a node is within noise of beating best

# The unbiased metric's noise is not the validation metric's noise. The random-exposure set is
# 288,338 rows in which ~63% of users have no positive, so its GAUC rests on far fewer users and
# spreads wider across seeds. Borrowing SEED_STD (a validation-set figure) as the tolerance
# rejected every single candidate in runs/v2. Instead the baseline's own seed spread is measured
# -- it already runs all three seeds -- and the tolerance derived from it. 2.5 sigma is the same
# convention EPSILON uses.
UNBIASED_TOLERANCE_SIGMAS = 2.5
UNBIASED_TOLERANCE_DEFAULT = 0.003   # only if the baseline yields fewer than 2 usable seeds

# Failure diagnosis. "Under-trained" means the model scores worse on the data it was fitted to
# than the baseline scores on data it has never seen -- an unambiguous sign that optimisation
# failed rather than that the hypothesis was wrong, and the two call for opposite responses.
# Anchoring on the baseline's *validation* score rather than a fixed margin below its training
# score is what keeps this honest: replaying runs/v2, a 0.02 margin also flagged nodes 10 and
# 15 (train 0.6699 and 0.6631), and node 15 was the best node in the entire run.
UNDER_TRAINED_MARGIN = 0.0
# A train-val gap this multiple of the baseline's gap is overfitting rather than a bad idea.
OVERFIT_GAP_RATIO = 1.5

# Execution sandbox. Raised alongside library access: the numpy FM trains in ~40s, but a CPU
# torch or GBDT run is a different order of magnitude, and runs/v2 already saw a 559s iteration.
EXEC_TIMEOUT_S = 1800
# Only enforced on Linux. macOS refuses to lower RLIMIT_AS from its inherited infinite soft
# limit, so executor._limit_memory silently no-ops there -- see the note in that function.
EXEC_MEMORY_CAP_GB = 24
STDOUT_TAIL_CHARS = 1500

# LLM access through AWS Bedrock.
# .env is the single source of truth for all four of these. No fallbacks, no defaults:
# an unset value is an error at call time, not something quietly guessed here.
#   AWS_BEARER_TOKEN     -- Bedrock API key
#   AWS_REGION           -- e.g. ap-southeast-1
#   AGENT_*_MODEL        -- one inference profile id per role, listed below
BEDROCK_API_KEY = _env('AWS_BEARER_TOKEN')
AWS_REGION = _env('AWS_REGION')

# One model per role, each set independently in .env so a role can be tuned without
# disturbing the others. These are all Claude models differing in capability rather than task
# specialisation, so choose by how hard the role's job is.
#
# Measured inputs to those choices:
#   - Haiku 4.5 as the coder emitted reasoning prose inside the code fence and produced a file
#     that would not parse. Roles that write a whole training file want a stronger model.
#   - Prompt caching needs a prefix above a model-specific minimum: 1024 tokens on Sonnet 4.6,
#     4096 on Haiku 4.5. The ~2.5K-token task description therefore caches on the former and
#     silently does not on the latter.
PLANNER_MODEL = _env('AGENT_PLANNER_MODEL')
BASELINE_MODEL = _env('AGENT_BASELINE_MODEL')
EDA_MODEL = _env('AGENT_EDA_MODEL')
CODER_MODEL = _env('AGENT_CODER_MODEL')
REVIEWER_MODEL = _env('AGENT_REVIEWER_MODEL')
DEBUGGER_MODEL = _env('AGENT_DEBUGGER_MODEL')
# Second debug attempt on the same branch: the cheap fix has already failed once.
DEBUGGER_RETRY_MODEL = _env('AGENT_DEBUGGER_RETRY_MODEL')

MAX_OUTPUT_TOKENS = 16000
LLM_MAX_RETRIES = 5
LLM_BACKOFF_BASE_S = 2.0

# Literature search. OpenAlex is used rather than arXiv or Semantic Scholar: both of those
# return HTTP 429 from this network, OpenAlex does not. The mailto puts requests in its
# polite pool. Caps here exist to bound context growth -- the whole conversation is resent on
# every tool round-trip, so papers are the most expensive thing the Planner can ask for.
RESEARCH_ENABLED = True
OPENALEX_MAILTO = _env('OPENALEX_MAILTO', 'kuairand-agent@example.com')
RESEARCH_CACHE_DIR = os.path.join(REPO_ROOT, '.cache', 'openalex')
MAX_SEARCHES_PER_CALL = 2
PAPERS_PER_SEARCH = 3
ABSTRACT_CHARS = 300
RESEARCH_TIMEOUT_S = 25

# Files the agent must never modify.
PROTECTED_FILES = ['evaluate.py', 'submit.py', 'data.py', 'baseline.py']
