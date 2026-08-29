"""Agent-wide constants. Everything tunable lives here so no magic numbers hide in the loop."""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
#   AGENT_STRONG_MODEL   -- inference profile id used for drafts and hard debugs
#   AGENT_FAST_MODEL     -- inference profile id used for routine improve/debug
BEDROCK_API_KEY = _env('AWS_BEARER_TOKEN')
AWS_REGION = _env('AWS_REGION')
STRONG_MODEL = _env('AGENT_STRONG_MODEL')
FAST_MODEL = _env('AGENT_FAST_MODEL')

MAX_OUTPUT_TOKENS = 16000
LLM_MAX_RETRIES = 5
LLM_BACKOFF_BASE_S = 2.0

# Files the agent must never modify.
PROTECTED_FILES = ['evaluate.py', 'submit.py', 'data.py', 'baseline.py']
