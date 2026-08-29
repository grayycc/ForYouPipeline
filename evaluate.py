"""
KuaiRand-Pure official evaluation script -- all conventions are fixed here, do not change them.

Task              : within-user ranking (within-user ranking over logged impressions)
Relevance label   : long_view (native column, 0/1)
Metrics           : GAUC, nDCG@5  (primary score = the average of the two)
Ranking scope     : each user only ranks their own impressions in the evaluation set, no full-corpus retrieval
Zero-positive users: nDCG counts as 0.0 and is included in the average (consistent with CWM)
                     GAUC only counts users with 0 < #positives < #impressions, weighted by positive count
nDCG gain         : (2^rel - 1), equivalent to identity under binary labels
Data split        : train 20220408-20220421 / valid 20220422-20220428 / test 20220429-20220508
"""
import math, collections

def auc(labels, scores):
    """Mann-Whitney U, with tie correction, equivalent to sklearn.metrics.roc_auc_score."""
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    npos = sum(l for _, l in pairs)
    nneg = len(pairs) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    srank = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (srank - npos * (npos + 1) / 2.0) / (npos * nneg)

def ndcg_at_k(labels, k):
    """labels is already sorted by predicted score in descending order."""
    disc = [math.log2(i + 2) for i in range(k)]
    dcg = sum(((2 ** t) - 1) / disc[i] for i, t in enumerate(labels[:k]))
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum(((2 ** t) - 1) / disc[i] for i, t in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg

def evaluate(user_ids, labels, scores, k=5):
    """Returns {'GAUC':..., 'nDCG@5':..., 'primary':...}. primary = the average of the two, used for ranking."""
    byu = collections.defaultdict(list)
    for u, y, s in zip(user_ids, labels, scores):
        byu[u].append((s, y))
    gnum = gden = 0.0
    nd = []
    for u, lst in byu.items():
        lst.sort(key=lambda x: -x[0])
        labs = [y for _, y in lst]
        npos = sum(labs)
        if 0 < npos < len(labs):
            gnum += npos * auc(labs, [s for s, _ in lst])
            gden += npos
        nd.append(ndcg_at_k(labs, k))
    gauc = gnum / gden if gden else 0.5
    ndcg = sum(nd) / len(nd) if nd else 0.0
    return {'GAUC': gauc, f'nDCG@{k}': ndcg, 'primary': (gauc + ndcg) / 2.0,
            'users': len(byu), 'rows': len(labels)}
