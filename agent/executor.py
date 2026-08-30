"""Sandboxed execution of LLM-generated solutions + deterministic metric parsing.

Design notes:
  * metrics are regex-parsed from stdout, never read by an LLM call, which is free efficiency
  * every failure mode collapses to "buggy", never propagates into the loop
  * tracebacks are trimmed before going back to the LLM: shorter, more relevant, cheaper
"""
import os
import re
import subprocess
import sys
import time

from . import config

FENCE_RE = re.compile(r'^\s*```(?:python|py)?\s*\n(.*?)\n\s*```\s*$', re.DOTALL)
METRIC_RES = {
    'train_primary': re.compile(r'^TRAIN_PRIMARY\s*=\s*([-+0-9.eE]+)\s*$', re.MULTILINE),
    'val_gauc': re.compile(r'^VAL_GAUC\s*=\s*([-+0-9.eE]+)\s*$', re.MULTILINE),
    'val_ndcg5': re.compile(r'^VAL_NDCG5\s*=\s*([-+0-9.eE]+)\s*$', re.MULTILINE),
    'val_primary': re.compile(r'^VAL_PRIMARY\s*=\s*([-+0-9.eE]+)\s*$', re.MULTILINE),
    'unbiased_val_primary': re.compile(r'^UNBIASED_PRIMARY\s*=\s*([-+0-9.eE]+)\s*$', re.MULTILINE),
}
EXC_RE = re.compile(r'^(\w+(?:Error|Exception|Interrupt)):', re.MULTILINE)

# subprocess reports a signal death as a negative return code; a shell would report 128+signal.
# SIGSEGV is 11, SIGBUS 10, SIGABRT 6.
SEGFAULT_CODES = (-11, 139, -10, 138, -6, 134)
SEGFAULT_HINT = (
    'The process died on a signal (segfault/abort) rather than raising, so there is no Python '
    'traceback to read.\n'
    'By far the most likely cause on this machine: `torch` was imported in the same script as '
    '`lightgbm` or `xgboost`. Those libraries each bundle their own OpenMP runtime and crash '
    'when loaded together -- verified, and neither KMP_DUPLICATE_LIB_OK nor OMP_NUM_THREADS=1 '
    'avoids it.\n'
    'Fix: use torch OR the gradient-boosting libraries in a single solution, never both. '
    '(scikit-learn, scipy, pandas and numpy are safe alongside either.)'
)


def strip_fences(text: str) -> str:
    """LLMs wrap code in markdown fences; unstripped, every iteration dies on a syntax error."""
    text = text.strip()
    m = FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    # fall back: if the body still contains a fence, take the largest fenced block
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)\n```', text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    # last resort: unbalanced or truncated fences -- drop the fence lines themselves so the
    # remainder is still valid Python rather than a guaranteed SyntaxError on line 1.
    if '```' in text:
        return '\n'.join(ln for ln in text.splitlines()
                         if not ln.strip().startswith('```')).strip()
    return text


def trim_traceback(stderr: str, max_chars: int = config.STDOUT_TAIL_CHARS) -> str:
    """Drop absolute paths and our own frames so the model sees the user code's error."""
    lines = []
    for ln in stderr.splitlines():
        if '/agent/' in ln or 'site-packages' in ln:
            continue
        lines.append(ln.replace(config.REPO_ROOT + os.sep, ''))
    out = '\n'.join(lines).strip()
    return out[-max_chars:]


def parse_metrics(stdout: str) -> dict:
    out = {}
    for key, rx in METRIC_RES.items():
        m = rx.findall(stdout)
        if m:
            try:
                out[key] = float(m[-1])   # last occurrence wins
            except ValueError:
                pass
    return out


def _limit_memory():
    """Best-effort address-space cap. Not supported everywhere, so never fatal.

    Measured on this machine (macOS 15, arm64): setrlimit(RLIMIT_AS, ...) raises
    `ValueError: current limit exceeds maximum limit` for every value, because the inherited
    soft limit is already RLIM_INFINITY and macOS will not lower it here. The except below has
    therefore always swallowed it and EXEC_MEMORY_CAP_GB has never actually bounded anything on
    darwin. Left in place because it does work on Linux, but do not rely on it as a guard.
    """
    try:
        import resource
        cap = config.EXEC_MEMORY_CAP_GB * 1024 ** 3
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except Exception:
        pass


def run_solution(code_path: str, out_dir: str, seed: int = 0,
                 timeout: int = config.EXEC_TIMEOUT_S,
                 require_metrics: bool = True,
                 extra_args: list = None) -> dict:
    """Run one generated solution in a subprocess. Returns a result dict; never raises.

    `require_metrics=False` is for the EDA pass, which produces findings rather than a score:
    a clean exit is success there, and demanding VAL_PRIMARY would mark it buggy and send it
    to the debugger.
    """
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, code_path,
           '--data_dir', config.DATA_DIR,
           '--out_dir', out_dir,
           '--seed', str(seed)] + list(extra_args or [])
    # the solution lives in its node dir, so sys.path[0] is *not* the repo root:
    # PYTHONPATH is what lets it `from data import load` (kit/) and import our modules.
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [config.KIT_DIR, config.REPO_ROOT, env.get('PYTHONPATH', '')]).rstrip(os.pathsep)
    t0 = time.monotonic()
    try:
        p = subprocess.run(
            cmd, cwd=config.REPO_ROOT, capture_output=True, text=True, env=env,
            timeout=timeout, preexec_fn=_limit_memory if os.name == 'posix' else None,
        )
        stdout, stderr, rc, timed_out = p.stdout, p.stderr, p.returncode, False
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or '')
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or '')
        rc, timed_out = -1, True
    except Exception as e:                                  # launch failure
        stdout, stderr, rc, timed_out = '', f'{type(e).__name__}: {e}', -1, False

    elapsed = time.monotonic() - t0
    metrics = parse_metrics(stdout)

    if timed_out:
        buggy, reason, exc = True, f'timeout>{timeout}s', 'Timeout'
    elif rc in SEGFAULT_CODES:
        # A segfault kills the interpreter before it can raise, so stderr comes back empty and
        # the debugger role has nothing to work from. On this machine the overwhelmingly likely
        # cause is the OpenMP clash below, so say so rather than handing back a bare exit code.
        buggy, reason, exc = True, 'segfault', 'Segfault'
        stderr = (stderr + '\n' + SEGFAULT_HINT).strip()
    elif rc != 0:
        m = EXC_RE.findall(stderr)
        buggy, reason, exc = True, 'exception', (m[-1] if m else 'NonZeroExit')
    elif require_metrics and 'val_primary' not in metrics:
        buggy, reason, exc = True, 'no_metric_printed', None
    else:
        buggy, reason, exc = False, '', None

    return {
        'is_buggy': buggy, 'buggy_reason': reason, 'exception_type': exc,
        'metrics': metrics, 'exec_time': elapsed, 'returncode': rc,
        'stdout_tail': stdout[-config.STDOUT_TAIL_CHARS:],
        'stderr_tail': trim_traceback(stderr),
    }


def check_submission(path: str, split: str) -> tuple:
    """Run the official validator. A node whose submission fails is buggy no matter its metric."""
    if not os.path.exists(path):
        return False, f'{os.path.basename(path)} was not created'
    try:
        p = subprocess.run(
            [sys.executable, os.path.join(config.KIT_DIR, 'submit.py'),
             '--check', '--split', split, '--data_dir', config.DATA_DIR, path],
            cwd=config.KIT_DIR, capture_output=True, text=True, timeout=600,
        )
        return p.returncode == 0, trim_traceback(p.stderr or p.stdout, 600)
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'
