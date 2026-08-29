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


DATA_DIR = os.path.join(REPO_ROOT, 'data', 'KuaiRand-Pure', 'data')
RUNS_DIR = os.path.join(REPO_ROOT, 'runs')

# Budget, from the problem statement.
MAX_ITERATIONS = 50               # hard cap per benchmark run
WALL_CLOCK_CEILING_S = 6 * 3600   # backstop

# Convergence. The README reports a seed std of 0.0008, so eps is about 2.5 sigma.
EPSILON = 0.002
CONVERGENCE_N = 3

# Reference scores to measure against.
BASELINE_VALID_PRIMARY = 0.6016
BASELINE_TEST_PRIMARY = 0.5946
ORACLE_PRIMARY = 0.8645
SEED_STD = 0.0008

# Search policy.
MIN_DRAFTS = 3                    # distinct fresh solutions before improve-only
DEBUG_PROBABILITY = 0.5           # chance of debugging a buggy leaf when one exists
MAX_DEBUG_DEPTH = 2               # abandon a branch after 2 failed fixes (playbook: not 3)

# Seed averaging, only spent on candidates that already look promising.
CONFIRM_SEEDS = [0, 1, 2]
CONFIRM_TRIGGER = EPSILON / 2     # re-run with more seeds once a node is within noise of beating best

# Execution sandbox.
EXEC_TIMEOUT_S = 900              # 15 min; baseline is ~40s, so this is generous
EXEC_MEMORY_CAP_GB = 12
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
