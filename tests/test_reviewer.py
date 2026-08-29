"""The leakage reviewer must not block legitimate feature engineering.

This exists because an earlier prompt flagged train-only count encoding as leakage. That is
the playbook's second-highest-value technique, and a LEAK verdict blocks a node from ever
becoming the submission -- so a false positive here costs more than a missed one elsewhere.

Makes real API calls. Run deliberately, not in a tight loop.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import config
from agent.llm import LLMClient
from agent.roles import reviewer
from agent.spec import ExperimentSpec

CASES = [
    ('train-only author statistics', 'CLEAN', '''
splits = load(args.data_dir)
author_rate = {}
for row in splits['train']:                       # training split only
    author_rate.setdefault(row[3], []).append(row[6])
author_rate = {a: sum(v) / len(v) for a, v in author_rate.items()}
def feat(row): return author_rate.get(row[3], 0.21)
'''),
    ('expanding window excluding current row', 'CLEAN', '''
by_date = sorted(splits['train'], key=lambda r: r[0])
running = {}
for row in by_date:
    f = running.get(row[2], 0.21)                 # read before update
    feats.append(f)
    running[row[2]] = 0.9 * running.get(row[2], 0.21) + 0.1 * row[6]
'''),
    ('fitting model parameters on train labels', 'CLEAN', '''
for epoch in range(20):
    for xb, yb in batches(Xtr, ytr):
        model.step(xb, yb)                        # training, not leakage
'''),
    ('global statistics over all splits', 'LEAK', '''
allrows = splits['train'] + splits['valid'] + splits['test']
pop = {}
for row in allrows:                               # includes evaluation rows
    pop.setdefault(row[2], []).append(row[6])
'''),
    ('target encoding fitted on predicted rows', 'LEAK', '''
for row in splits['valid']:
    rate[row[2]] = rate.get(row[2], []) + [row[6]]    # uses the validation label
score = [rate[r[2]] for r in splits['valid']]
'''),
]


def run(model=None, verbose=True):
    model = model or config.REVIEWER_MODEL
    config.REVIEWER_MODEL = model
    llm = LLMClient()
    spec = ExperimentSpec(proposed_change='add a feature', risks={'leakage': 'n/a'})
    wrong = []
    for name, expected, code in CASES:
        verdict, reason, _, _ = reviewer.run(llm, code, spec)
        ok = verdict == expected
        if not ok:
            wrong.append((name, expected, verdict, reason))
        if verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  want {expected:<5} got {verdict:<5}  {name}")
    return wrong


if __name__ == '__main__':
    print(f'reviewer model: {config.REVIEWER_MODEL}')
    wrong = run()
    if wrong:
        print(f'\n{len(wrong)} incorrect:')
        for name, exp, got, reason in wrong:
            print(f'  {name}: wanted {exp}, got {got} -- {reason[:150]}')
        raise SystemExit(1)
    print(f'\nall {len(CASES)} reviewer cases correct')
