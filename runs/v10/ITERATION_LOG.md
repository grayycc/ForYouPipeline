# Iteration log — run `v10`

8 nodes. Each code diff below is computed from the actual files this run wrote, against the actual parent it branched from -- not a description.

## Node 0 — `baseline`

**Hypothesis:**

I will reproduce the FM baseline exactly by importing the `FM` class from `baseline.py` and writing my own training loop that matches the reference implementation: 5 categorical fields (user_id, video_id, author_id, tab, dur_bucket), embedding dim k=16, Adam lr=0.001, batch size 8192, pointwise BCE loss, early stopping on validation primary with patience 4. I expect to land within the noise floor of 0.6016 (i.e., between ~0.599 and ~0.604). This confirms the pipeline works end-to-end before any modifications.

**Metrics:**

| train_primary | val_gauc | val_ndcg5 | val_primary | unbiased | diagnosis |
|---|---|---|---|---|---|
| 0.6918 | 0.6671 | 0.5358 | **0.6015** | 0.3648 | noise |

**Accepted:** yes — became the new best

**Code diff:** none — fresh file, no parent (146 lines). See `nodes/node_0/solution.py`.

## Node 1 — `eda`

**Hypothesis:**

I want to understand the data distribution to guide feature engineering and modeling choices. Specifically: (1) how many users have training history vs. are cold-start in validation, (2) the distribution of label rates per user/video, (3) what interaction signals might be useful for time-based features, and (4) whether there are patterns in the raw log columns that could inform feature construction without leakage.

**Result:** no score by design (eda)

**Code diff:** none — fresh file, no parent (133 lines). See `nodes/node_1/solution.py`.

## Node 2 — `improve` (parent: node 0)

**Hypothesis:**

Averaging within-user fractional ranks across 5 independently-seeded FM models will improve validation primary by ~0.0016 over a single-seed model, because rank averaging reduces variance in the ordering decisions that directly determine GAUC and nDCG@5

**Metrics:**

| train_primary | val_gauc | val_ndcg5 | val_primary | unbiased | diagnosis |
|---|---|---|---|---|---|
| 0.6863 | 0.6691 | 0.5366 | **0.6028** | 0.3653 | noise |

**Accepted:** yes — became the new best
**Leakage review:** CLEAN

**Code diff** (+121/-37 lines vs. node 0):

```diff
--- node_0/solution.py
+++ node_2/solution.py
@@ -1,8 +1,8 @@
 #!/usr/bin/env python3
 """
-Iteration 0: Reproduce the FM baseline.
-Trains an FM over 5 categorical fields, evaluates on validation,
-writes submissions for valid and test.
+Iteration 1: Seed ensemble with within-user rank averaging.
+Train 5 FM models with seeds 0-4, convert each model's scores to within-user
+fractional ranks, average ranks across models, use averaged rank as final score.
 """
 
 import argparse
@@ -36,11 +36,42 @@
 X_va, y_va, u_va = enc['valid']
 X_te, y_te, u_te = enc['test']
 
-# ── Training (train mode) ─────────────────────────────────────────────────────
+# ── Hyperparameters ───────────────────────────────────────────────────────────
 BATCH = 8192
 PATIENCE = 4
 MAX_EPOCHS = 50
-
+N_SEEDS = 5
+
+# ── Helper: within-user fractional ranks ──────────────────────────────────────
+def within_user_ranks(users, scores):
+    """
+    For each user, convert raw scores to fractional ranks in [0, 1].
+    Rank 0 = lowest score, rank 1 = highest score (within that user's items).
+    Uses scipy-style fractional ranking to handle ties gracefully.
+    """
+    user_arr = np.array(users)
+    ranks = np.zeros(len(scores), dtype=np.float64)
+    unique_users = np.unique(user_arr)
+    for u in unique_users:
+        mask = user_arr == u
+        s = scores[mask]
+        n = mask.sum()
+        if n == 1:
+            ranks[mask] = 0.5
+            continue
+        # argsort of argsort gives rank (0-based), then normalize to [0,1]
+        order = np.argsort(s)
+        r = np.empty(n, dtype=np.float64)
+        r[order] = np.arange(n, dtype=np.float64)
+        # fractional rank in [0, 1]
+        if n > 1:
+            r = r / (n - 1)
+        else:
+            r = np.full(n, 0.5)
+        ranks[mask] = r
+    return ranks
+
+# ── Training helper ───────────────────────────────────────────────────────────
 def run_epoch(model, X, y, rng, batch_size=BATCH):
     """One full pass over data in random order, returns mean loss."""
     idx = rng.permutation(len(y))
@@ -51,9 +82,11 @@
         losses.append(loss)
     return float(np.mean(losses))
 
-if args.train_split == 'train':
-    rng = np.random.default_rng(args.seed)
-    model = FM(dim, k=16, lr=0.001, l2=1e-6, seed=args.seed)
+
+def train_single_model(seed, X_tr, y_tr, X_va, y_va, u_va, dim):
+    """Train one FM model with given seed, return best model weights and epoch count."""
+    rng = np.random.default_rng(seed)
+    model = FM(dim, k=16, lr=0.001, l2=1e-6, seed=seed)
 
     best_val = -1.0
     best_weights = None
@@ -67,12 +100,11 @@
         val_res = evaluate(u_va, y_va, val_scores)
         val_primary = val_res['primary']
 
-        print(f"Epoch {epoch:02d}  tr_loss={tr_loss:.4f}  "
+        print(f"  Seed {seed} Epoch {epoch:02d}  tr_loss={tr_loss:.4f}  "
               f"val_primary={val_primary:.4f}", flush=True)
 
         if val_primary > best_val + 1e-6:
             best_val = val_primary
-            # Save weights
             best_weights = (
                 model.V.copy(),
                 model.W.copy(),
@@ -83,7 +115,7 @@
         else:
             patience_left -= 1
             if patience_left <= 0:
-                print(f"Early stop at epoch {epoch}, best was {best_epoch}",
+                print(f"  Seed {seed} early stop at epoch {epoch}, best was {best_epoch}",
                       flush=True)
                 break
 
@@ -92,19 +124,62 @@
     model.W[:] = best_weights[1]
     model.b = best_weights[2]
 
-    # ── Evaluate ─────────────────────────────────────────────────────────────
-    tr_scores = model.predict(X_tr)
-    tr_res = evaluate(u_tr, y_tr, tr_scores)
-
-    va_scores = model.predict(X_va)
-    va_res = evaluate(u_va, y_va, va_scores)
-
-    # Unbiased evaluation
+    return model, best_epoch
+
+
+if args.train_split == 'train':
+    # ── Train N_SEEDS models and collect scores ────────────────────────────────
+    va_rank_sum = np.zeros(len(y_va), dtype=np.float64)
+    te_rank_sum = np.zeros(len(y_te), dtype=np.float64)
+    tr_rank_sum = np.zeros(len(y_tr), dtype=np.float64)
+    rand_rank_sum = None
+    u_rand_arr = None
+
+    # Load random data once for unbiased evaluation
     rand_rows = load_random_valid(args.data_dir)
     X_rand, y_rand, u_rand, _ = encode_like_train(splits['train'], rand_rows)
+    rand_rank_sum = np.zeros(len(y_rand), dtype=np.float64)
+
+    best_epochs = []
+
+    for s in range(N_SEEDS):
+        print(f"\n=== Training seed {s} ===", flush=True)
+        model, best_epoch = train_single_model(s, X_tr, y_tr, X_va, y_va, u_va, dim)
+        best_epochs.append(best_epoch)
+
+        # Compute scores for each split
+        va_scores = model.predict(X_va)
+        te_scores = model.predict(X_te)
+        tr_scores = model.predict(X_tr)
+        rand_scores = model.predict(X_rand)
+
+        # Convert to within-user fractional ranks
+        va_ranks = within_user_ranks(u_va, va_scores)
+        te_ranks = within_user_ranks(u_te, te_scores)
+        tr_ranks = within_user_ranks(u_tr, tr_scores)
+        rand_ranks = within_user_ranks(u_rand, rand_scores)
+
+        va_rank_sum += va_ranks
+        te_rank_sum += te_ranks
+        tr_rank_sum += tr_ranks
+        rand_rank_sum += rand_ranks
+
+    # Average ranks
+    va_final = va_rank_sum / N_SEEDS
+    te_final = te_rank_sum / N_SEEDS
+    tr_final = tr_rank_sum / N_SEEDS
+    rand_final = rand_rank_sum / N_SEEDS
+
+    print(f"\nBest epochs across seeds: {best_epochs}", flush=True)
+
+    # ── Evaluate ──────────────────────────────────────────────────────────────
+    tr_res = evaluate(u_tr, y_tr, tr_final)
+    va_res = evaluate(u_va, y_va, va_final)
+
+    # Unbiased evaluation using averaged ranks
     unbiased = unbiased_primary(
         args.data_dir, splits['train'],
-        lambda rows: model.predict(X_rand)
+        lambda rows: rand_final
     )
 
     print(f"TRAIN_PRIMARY={tr_res['primary']:.4f}")
@@ -116,31 +191,40 @@
     # ── Write submissions ──────────────────────────────────────────────────────
     submit.write_submission(
         os.path.join(args.out_dir, 'submission_valid.csv'),
-        splits['valid'], va_scores
+        splits['valid'], va_final
     )
-    te_scores = model.predict(X_te)
     submit.write_submission(
         os.path.join(args.out_dir, 'submission_test.csv'),
-        splits['test'], te_scores
+        splits['test'], te_final
     )
 
 else:
-    # train+valid mode: combine train and valid, train for best_epoch epochs
-    # Use a fixed epoch count (15 epochs is typically where baseline converges)
+    # train+valid mode: combine train and valid, train for average best_epoch epochs
+    # Use fixed epoch count based on typical convergence
     X_all = np.concatenate([X_tr, X_va], axis=0)
     y_all = np.concatenate([y_tr, y_va], axis=0)
-
-    FIXED_EPOCHS = 15  # conservative fixed schedule
-
-    rng = np.random.default_rng(args.seed)
-    model = FM(dim, k=16, lr=0.001, l2=1e-6, seed=args.seed)
-
-    for epoch in range(1, FIXED_EPOCHS + 1):
-        tr_loss = run_epoch(model, X_all, y_all, rng)
-        print(f"Epoch {epoch:02d}  tr_loss={tr_loss:.4f}", flush=True)
-
-    te_scores = model.predict(X_te)
+    u_all = list(u_tr) + list(u_va)
+
+    FIXED_EPOCHS = 10  # conservative fixed schedule for train+valid
+
+    te_rank_sum = np.zeros(len(y_te), dtype=np.float64)
+
+    for s in range(N_SEEDS):
+        print(f"\n=== Training seed {s} (train+valid) ===", flush=True)
+        rng = np.random.default_rng(s)
+        model = FM(dim, k=16, lr=0.001, l2=1e-6, seed=s)
+
+        for epoch in range(1, FIXED_EPOCHS + 1):
+            tr_loss = run_epoch(model, X_all, y_all, rng)
+            print(f"  Seed {s} Epoch {epoch:02d}  tr_loss={tr_loss:.4f}", flush=True)
+
+        te_scores = model.predict(X_te)
+        te_ranks = within_user_ranks(u_te, te_scores)
+        te_rank_sum += te_ranks
+
+    te_final = te_rank_sum / N_SEEDS
+
     submit.write_submission(
         os.path.join(args.out_dir, 'submission_test.csv'),
-        splits['test'], te_scores
+        splits['test'], te_final
     )
```

## Node 3 — `improve` (parent: node 2)

**Hypothesis:**

Adding Bayesian-smoothed per-video and per-author historical long_view rates (bucketed into categorical features) to the FM on top of the 5-seed rank ensemble will increase validation primary by ~+0.0008, pushing the combined system to ~0.6036+

**Metrics:**

| train_primary | val_gauc | val_ndcg5 | val_primary | unbiased | diagnosis |
|---|---|---|---|---|---|
| 0.6710 | 0.6697 | 0.5371 | **0.6034** | 0.3642 | noise |

**Accepted:** yes — became the new best
**Leakage review:** CLEAN

**Code diff** (+162/-27 lines vs. node 2):

```diff
--- node_2/solution.py
+++ node_3/solution.py
@@ -1,8 +1,12 @@
 #!/usr/bin/env python3
 """
-Iteration 1: Seed ensemble with within-user rank averaging.
-Train 5 FM models with seeds 0-4, convert each model's scores to within-user
-fractional ranks, average ranks across models, use averaged rank as final score.
+Iteration 2: Add Bayesian-smoothed per-video and per-author long_view rates (bucketed)
+to the 5-seed FM ensemble with within-user rank averaging.
+
+Changes from iteration 1:
+- Compute Bayesian-smoothed long_view rate for each video and author from training data only
+- Bucket each into 10 equal-frequency bins
+- Add these 2 new categorical features to FM input (7 features total instead of 5)
 """
 
 import argparse
@@ -22,7 +26,7 @@
 os.makedirs(args.out_dir, exist_ok=True)
 
 # ── Imports ───────────────────────────────────────────────────────────────────
-from data import load, encode
+from data import load, encode, FIELDS
 from baseline import FM
 from evaluate import evaluate
 import submit
@@ -32,9 +36,138 @@
 splits = load(args.data_dir)
 
 enc, dim = encode(splits)
-X_tr, y_tr, u_tr = enc['train']
-X_va, y_va, u_va = enc['valid']
-X_te, y_te, u_te = enc['test']
+X_tr_base, y_tr, u_tr = enc['train']
+X_va_base, y_va, u_va = enc['valid']
+X_te_base, y_te, u_te = enc['test']
+
+# ── Compute Bayesian-smoothed CTR features from training data only ────────────
+# Fields: user_id=0, video_id=1, author_id=2, tab=3, dur_bucket=4, long_view=5 in raw rows
+# We use splits['train'] which has tuples: (date, user_id, video_id, author_id, tab, duration_ms, long_view)
+
+GLOBAL_MEAN_PRIOR_STRENGTH = 50  # pseudo-count for Bayesian smoothing
+
+def compute_bayesian_ctr(train_rows, key_idx, label_idx=6):
+    """
+    Compute Bayesian-smoothed long_view rate keyed by field at key_idx.
+    Returns a dict: key -> smoothed_rate
+    key_idx: index into the row tuple (1=user_id, 2=video_id, 3=author_id)
+    label_idx: index of long_view label (6 in load() tuples)
+    """
+    counts = {}  # key -> [total, positives]
+    for row in train_rows:
+        key = row[key_idx]
+        label = row[label_idx]
+        if key not in counts:
+            counts[key] = [0, 0]
+        counts[key][0] += 1
+        counts[key][1] += label
+
+    # Global rate (prior)
+    total_all = sum(v[0] for v in counts.values())
+    pos_all = sum(v[1] for v in counts.values())
+    global_rate = pos_all / total_all if total_all > 0 else 0.5
+
+    # Bayesian smoothing: rate = (positives + alpha * global_rate) / (total + alpha)
+    alpha = GLOBAL_MEAN_PRIOR_STRENGTH
+    smoothed = {}
+    for key, (total, pos) in counts.items():
+        smoothed[key] = (pos + alpha * global_rate) / (total + alpha)
+
+    return smoothed, global_rate
+
+
+def make_bucket_feature(rows, ctr_map, global_rate, key_idx, n_bins=10):
+    """
+    For each row, look up the Bayesian-smoothed CTR and bucket it.
+    Returns integer array of bucket indices (0 to n_bins-1).
+    Unknown keys get global_rate.
+    """
+    rates = np.array([
+        ctr_map.get(row[key_idx], global_rate)
+        for row in rows
+    ], dtype=np.float64)
+    return rates
+
+
+def compute_buckets_from_rates(train_rates, query_rates, n_bins=10):
+    """
+    Use quantile edges from training rates to bucket query rates.
+    Returns integer bucket indices (0 to n_bins-1).
+    """
+    # Compute quantile edges from training rates
+    quantiles = np.linspace(0, 100, n_bins + 1)
+    edges = np.percentile(train_rates, quantiles)
+    # Ensure edges are unique; if not, use unique edges
+    edges = np.unique(edges)
+    # Bucket query rates using these edges
+    # np.searchsorted: values below first edge -> 0, above last -> len(edges)
+    buckets = np.searchsorted(edges[1:-1], query_rates, side='right')
+    # Clip to [0, n_bins-1]
+    buckets = np.clip(buckets, 0, n_bins - 1)
+    return buckets.astype(np.int32)
+
+
+# Compute CTR maps from training data only
+train_rows = splits['train']
+valid_rows = splits['valid']
+test_rows = splits['test']
+
+print("Computing video CTR map...", flush=True)
+video_ctr_map, video_global_rate = compute_bayesian_ctr(train_rows, key_idx=2)  # video_id at idx 2
+print(f"  {len(video_ctr_map)} videos, global rate={video_global_rate:.4f}", flush=True)
+
+print("Computing author CTR map...", flush=True)
+author_ctr_map, author_global_rate = compute_bayesian_ctr(train_rows, key_idx=3)  # author_id at idx 3
+print(f"  {len(author_ctr_map)} authors, global rate={author_global_rate:.4f}", flush=True)
+
+# Get rates for all splits
+N_BINS = 10
+
+train_video_rates = make_bucket_feature(train_rows, video_ctr_map, video_global_rate, key_idx=2)
+train_author_rates = make_bucket_feature(train_rows, author_ctr_map, author_global_rate, key_idx=3)
+
+valid_video_rates = make_bucket_feature(valid_rows, video_ctr_map, video_global_rate, key_idx=2)
+valid_author_rates = make_bucket_feature(valid_rows, author_ctr_map, author_global_rate, key_idx=3)
+
+test_video_rates = make_bucket_feature(test_rows, video_ctr_map, video_global_rate, key_idx=2)
+test_author_rates = make_bucket_feature(test_rows, author_ctr_map, author_global_rate, key_idx=3)
+
+# Compute bucket assignments using training quantiles
+train_video_buckets = compute_buckets_from_rates(train_video_rates, train_video_rates, N_BINS)
+train_author_buckets = compute_buckets_from_rates(train_author_rates, train_author_rates, N_BINS)
+
+valid_video_buckets = compute_buckets_from_rates(train_video_rates, valid_video_rates, N_BINS)
+valid_author_buckets = compute_buckets_from_rates(train_author_rates, valid_author_rates, N_BINS)
+
+test_video_buckets = compute_buckets_from_rates(train_video_rates, test_video_rates, N_BINS)
+test_author_buckets = compute_buckets_from_rates(train_author_rates, test_author_rates, N_BINS)
+
+print(f"Video CTR bucket distribution (train): {np.bincount(train_video_buckets)}", flush=True)
+print(f"Author CTR bucket distribution (train): {np.bincount(train_author_buckets)}", flush=True)
+
+# ── Extend feature matrices with new CTR bucket features ─────────────────────
+# Add offset to new feature indices to avoid collision with existing features
+# Existing dim covers indices 0..dim-1
+# New features: video_ctr_bucket (0..N_BINS-1) and author_ctr_bucket (0..N_BINS-1)
+# They each get their own slot range after dim
+
+VIDEO_CTR_OFFSET = dim
+AUTHOR_CTR_OFFSET = dim + N_BINS
+NEW_DIM = dim + N_BINS + N_BINS
+
+print(f"Original embedding dim: {dim}, New dim: {NEW_DIM}", flush=True)
+
+def extend_features(X_base, video_buckets, author_buckets):
+    """Append video_ctr and author_ctr bucket columns to feature matrix."""
+    vid_col = (video_buckets + VIDEO_CTR_OFFSET).reshape(-1, 1).astype(np.int32)
+    auth_col = (author_buckets + AUTHOR_CTR_OFFSET).reshape(-1, 1).astype(np.int32)
+    return np.concatenate([X_base, vid_col, auth_col], axis=1)
+
+X_tr = extend_features(X_tr_base, train_video_buckets, train_author_buckets)
+X_va = extend_features(X_va_base, valid_video_buckets, valid_author_buckets)
+X_te = extend_features(X_te_base, test_video_buckets, test_author_buckets)
+
+print(f"Feature matrix shapes: train={X_tr.shape}, valid={X_va.shape}, test={X_te.shape}", flush=True)
 
 # ── Hyperparameters ───────────────────────────────────────────────────────────
 BATCH = 8192
@@ -47,7 +180,6 @@
     """
     For each user, convert raw scores to fractional ranks in [0, 1].
     Rank 0 = lowest score, rank 1 = highest score (within that user's items).
-    Uses scipy-style fractional ranking to handle ties gracefully.
     """
     user_arr = np.array(users)
     ranks = np.zeros(len(scores), dtype=np.float64)
@@ -59,15 +191,10 @@
         if n == 1:
             ranks[mask] = 0.5
             continue
-        # argsort of argsort gives rank (0-based), then normalize to [0,1]
         order = np.argsort(s)
         r = np.empty(n, dtype=np.float64)
         r[order] = np.arange(n, dtype=np.float64)
-        # fractional rank in [0, 1]
-        if n > 1:
-            r = r / (n - 1)
-        else:
-            r = np.full(n, 0.5)
+        r = r / (n - 1)
         ranks[mask] = r
     return ranks
 
@@ -83,10 +210,10 @@
     return float(np.mean(losses))
 
 
-def train_single_model(seed, X_tr, y_tr, X_va, y_va, u_va, dim):
-    """Train one FM model with given seed, return best model weights and epoch count."""
+def train_single_model(seed, X_tr, y_tr, X_va, y_va, u_va, new_dim):
+    """Train one FM model with given seed, return best model and epoch count."""
     rng = np.random.default_rng(seed)
-    model = FM(dim, k=16, lr=0.001, l2=1e-6, seed=seed)
+    model = FM(new_dim, k=16, lr=0.001, l2=1e-6, seed=seed)
 
     best_val = -1.0
     best_weights = None
@@ -128,23 +255,28 @@
 
 
 if args.train_split == 'train':
+    # ── Load random data for unbiased evaluation ──────────────────────────────
+    rand_rows = load_random_valid(args.data_dir)
+    X_rand_base, y_rand, u_rand, _ = encode_like_train(splits['train'], rand_rows)
+
+    # Compute CTR features for random rows (use same maps from training)
+    rand_video_rates = make_bucket_feature(rand_rows, video_ctr_map, video_global_rate, key_idx=2)
+    rand_author_rates = make_bucket_feature(rand_rows, author_ctr_map, author_global_rate, key_idx=3)
+    rand_video_buckets = compute_buckets_from_rates(train_video_rates, rand_video_rates, N_BINS)
+    rand_author_buckets = compute_buckets_from_rates(train_author_rates, rand_author_rates, N_BINS)
+    X_rand = extend_features(X_rand_base, rand_video_buckets, rand_author_buckets)
+
     # ── Train N_SEEDS models and collect scores ────────────────────────────────
     va_rank_sum = np.zeros(len(y_va), dtype=np.float64)
     te_rank_sum = np.zeros(len(y_te), dtype=np.float64)
     tr_rank_sum = np.zeros(len(y_tr), dtype=np.float64)
-    rand_rank_sum = None
-    u_rand_arr = None
-
-    # Load random data once for unbiased evaluation
-    rand_rows = load_random_valid(args.data_dir)
-    X_rand, y_rand, u_rand, _ = encode_like_train(splits['train'], rand_rows)
     rand_rank_sum = np.zeros(len(y_rand), dtype=np.float64)
 
     best_epochs = []
 
     for s in range(N_SEEDS):
         print(f"\n=== Training seed {s} ===", flush=True)
-        model, best_epoch = train_single_model(s, X_tr, y_tr, X_va, y_va, u_va, dim)
+        model, best_epoch = train_single_model(s, X_tr, y_tr, X_va, y_va, u_va, NEW_DIM)
         best_epochs.append(best_epoch)
 
         # Compute scores for each split
@@ -199,8 +331,11 @@
     )
 
 else:
-    # train+valid mode: combine train and valid, train for average best_epoch epochs
-    # Use fixed epoch count based on typical convergence
+    # train+valid mode: combine train and valid, train for fixed epoch count
+    # Need to recompute CTR features combining train+valid for proper features
+    # BUT: we must not recompute CTR maps from train+valid (would be leakage for test scoring)
+    # CTR maps remain from training only; just extend features for valid rows as done above
+
     X_all = np.concatenate([X_tr, X_va], axis=0)
     y_all = np.concatenate([y_tr, y_va], axis=0)
     u_all = list(u_tr) + list(u_va)
@@ -212,7 +347,7 @@
     for s in range(N_SEEDS):
         print(f"\n=== Training seed {s} (train+valid) ===", flush=True)
         rng = np.random.default_rng(s)
-        model = FM(dim, k=16, lr=0.001, l2=1e-6, seed=s)
+        model = FM(NEW_DIM, k=16, lr=0.001, l2=1e-6, seed=s)
 
         for epoch in range(1, FIXED_EPOCHS + 1):
             tr_loss = run_epoch(model, X_all, y_all, rng)
```

## Node 4 — `improve` (parent: node 3)

**Hypothesis:**

Replacing pointwise BCE with a BPR pairwise loss (sampling positive/negative pairs within each user per batch) will improve GAUC by directly optimizing the pairwise ranking objective, producing a gain of >0.002 over the current best

**Error:** `exception` (AttributeError)
**Recovery:** (none recorded)

**Code diff** (+215/-84 lines vs. node 3):

```diff
--- node_3/solution.py
+++ node_4/solution.py
@@ -1,12 +1,12 @@
 #!/usr/bin/env python3
 """
-Iteration 2: Add Bayesian-smoothed per-video and per-author long_view rates (bucketed)
-to the 5-seed FM ensemble with within-user rank averaging.
-
-Changes from iteration 1:
-- Compute Bayesian-smoothed long_view rate for each video and author from training data only
-- Bucket each into 10 equal-frequency bins
-- Add these 2 new categorical features to FM input (7 features total instead of 5)
+Iteration 3: Replace pointwise BCE with BPR pairwise loss.
+
+Changes from iteration 2:
+- Instead of pointwise BCE, sample within-user (positive, negative) pairs
+  and apply BPR loss = -log(sigmoid(score_pos - score_neg))
+- Uses FM's logits() to get scores, then manually computes gradients
+- Everything else identical: 5-seed ensemble, rank averaging, CTR features
 """
 
 import argparse
@@ -41,19 +41,10 @@
 X_te_base, y_te, u_te = enc['test']
 
 # ── Compute Bayesian-smoothed CTR features from training data only ────────────
-# Fields: user_id=0, video_id=1, author_id=2, tab=3, dur_bucket=4, long_view=5 in raw rows
-# We use splits['train'] which has tuples: (date, user_id, video_id, author_id, tab, duration_ms, long_view)
-
-GLOBAL_MEAN_PRIOR_STRENGTH = 50  # pseudo-count for Bayesian smoothing
+GLOBAL_MEAN_PRIOR_STRENGTH = 50
 
 def compute_bayesian_ctr(train_rows, key_idx, label_idx=6):
-    """
-    Compute Bayesian-smoothed long_view rate keyed by field at key_idx.
-    Returns a dict: key -> smoothed_rate
-    key_idx: index into the row tuple (1=user_id, 2=video_id, 3=author_id)
-    label_idx: index of long_view label (6 in load() tuples)
-    """
-    counts = {}  # key -> [total, positives]
+    counts = {}
     for row in train_rows:
         key = row[key_idx]
         label = row[label_idx]
@@ -62,12 +53,10 @@
         counts[key][0] += 1
         counts[key][1] += label
 
-    # Global rate (prior)
     total_all = sum(v[0] for v in counts.values())
     pos_all = sum(v[1] for v in counts.values())
     global_rate = pos_all / total_all if total_all > 0 else 0.5
 
-    # Bayesian smoothing: rate = (positives + alpha * global_rate) / (total + alpha)
     alpha = GLOBAL_MEAN_PRIOR_STRENGTH
     smoothed = {}
     for key, (total, pos) in counts.items():
@@ -76,12 +65,7 @@
     return smoothed, global_rate
 
 
-def make_bucket_feature(rows, ctr_map, global_rate, key_idx, n_bins=10):
-    """
-    For each row, look up the Bayesian-smoothed CTR and bucket it.
-    Returns integer array of bucket indices (0 to n_bins-1).
-    Unknown keys get global_rate.
-    """
+def make_bucket_feature(rows, ctr_map, global_rate, key_idx):
     rates = np.array([
         ctr_map.get(row[key_idx], global_rate)
         for row in rows
@@ -90,37 +74,26 @@
 
 
 def compute_buckets_from_rates(train_rates, query_rates, n_bins=10):
-    """
-    Use quantile edges from training rates to bucket query rates.
-    Returns integer bucket indices (0 to n_bins-1).
-    """
-    # Compute quantile edges from training rates
     quantiles = np.linspace(0, 100, n_bins + 1)
     edges = np.percentile(train_rates, quantiles)
-    # Ensure edges are unique; if not, use unique edges
     edges = np.unique(edges)
-    # Bucket query rates using these edges
-    # np.searchsorted: values below first edge -> 0, above last -> len(edges)
     buckets = np.searchsorted(edges[1:-1], query_rates, side='right')
-    # Clip to [0, n_bins-1]
     buckets = np.clip(buckets, 0, n_bins - 1)
     return buckets.astype(np.int32)
 
 
-# Compute CTR maps from training data only
 train_rows = splits['train']
 valid_rows = splits['valid']
 test_rows = splits['test']
 
 print("Computing video CTR map...", flush=True)
-video_ctr_map, video_global_rate = compute_bayesian_ctr(train_rows, key_idx=2)  # video_id at idx 2
+video_ctr_map, video_global_rate = compute_bayesian_ctr(train_rows, key_idx=2)
 print(f"  {len(video_ctr_map)} videos, global rate={video_global_rate:.4f}", flush=True)
 
 print("Computing author CTR map...", flush=True)
-author_ctr_map, author_global_rate = compute_bayesian_ctr(train_rows, key_idx=3)  # author_id at idx 3
+author_ctr_map, author_global_rate = compute_bayesian_ctr(train_rows, key_idx=3)
 print(f"  {len(author_ctr_map)} authors, global rate={author_global_rate:.4f}", flush=True)
 
-# Get rates for all splits
 N_BINS = 10
 
 train_video_rates = make_bucket_feature(train_rows, video_ctr_map, video_global_rate, key_idx=2)
@@ -132,7 +105,6 @@
 test_video_rates = make_bucket_feature(test_rows, video_ctr_map, video_global_rate, key_idx=2)
 test_author_rates = make_bucket_feature(test_rows, author_ctr_map, author_global_rate, key_idx=3)
 
-# Compute bucket assignments using training quantiles
 train_video_buckets = compute_buckets_from_rates(train_video_rates, train_video_rates, N_BINS)
 train_author_buckets = compute_buckets_from_rates(train_author_rates, train_author_rates, N_BINS)
 
@@ -141,15 +113,6 @@
 
 test_video_buckets = compute_buckets_from_rates(train_video_rates, test_video_rates, N_BINS)
 test_author_buckets = compute_buckets_from_rates(train_author_rates, test_author_rates, N_BINS)
-
-print(f"Video CTR bucket distribution (train): {np.bincount(train_video_buckets)}", flush=True)
-print(f"Author CTR bucket distribution (train): {np.bincount(train_author_buckets)}", flush=True)
-
-# ── Extend feature matrices with new CTR bucket features ─────────────────────
-# Add offset to new feature indices to avoid collision with existing features
-# Existing dim covers indices 0..dim-1
-# New features: video_ctr_bucket (0..N_BINS-1) and author_ctr_bucket (0..N_BINS-1)
-# They each get their own slot range after dim
 
 VIDEO_CTR_OFFSET = dim
 AUTHOR_CTR_OFFSET = dim + N_BINS
@@ -158,7 +121,6 @@
 print(f"Original embedding dim: {dim}, New dim: {NEW_DIM}", flush=True)
 
 def extend_features(X_base, video_buckets, author_buckets):
-    """Append video_ctr and author_ctr bucket columns to feature matrix."""
     vid_col = (video_buckets + VIDEO_CTR_OFFSET).reshape(-1, 1).astype(np.int32)
     auth_col = (author_buckets + AUTHOR_CTR_OFFSET).reshape(-1, 1).astype(np.int32)
     return np.concatenate([X_base, vid_col, auth_col], axis=1)
@@ -170,17 +132,13 @@
 print(f"Feature matrix shapes: train={X_tr.shape}, valid={X_va.shape}, test={X_te.shape}", flush=True)
 
 # ── Hyperparameters ───────────────────────────────────────────────────────────
-BATCH = 8192
+BATCH = 8192       # number of pairs per BPR batch
 PATIENCE = 4
 MAX_EPOCHS = 50
 N_SEEDS = 5
 
 # ── Helper: within-user fractional ranks ──────────────────────────────────────
 def within_user_ranks(users, scores):
-    """
-    For each user, convert raw scores to fractional ranks in [0, 1].
-    Rank 0 = lowest score, rank 1 = highest score (within that user's items).
-    """
     user_arr = np.array(users)
     ranks = np.zeros(len(scores), dtype=np.float64)
     unique_users = np.unique(user_arr)
@@ -198,36 +156,209 @@
         ranks[mask] = r
     return ranks
 
-# ── Training helper ───────────────────────────────────────────────────────────
-def run_epoch(model, X, y, rng, batch_size=BATCH):
-    """One full pass over data in random order, returns mean loss."""
-    idx = rng.permutation(len(y))
+
+# ── BPR training ──────────────────────────────────────────────────────────────
+def build_user_pos_neg_index(users, labels):
+    """
+    Build per-user lists of positive and negative indices.
+    Returns: user_pos (dict user -> list of row indices with label=1)
+             user_neg (dict user -> list of row indices with label=0)
+             eligible_users (list of users with both pos and neg)
+    """
+    user_pos = {}
+    user_neg = {}
+    for i, (u, lbl) in enumerate(zip(users, labels)):
+        if lbl > 0.5:
+            user_pos.setdefault(u, []).append(i)
+        else:
+            user_neg.setdefault(u, []).append(i)
+
+    eligible = [u for u in user_pos if u in user_neg]
+    return user_pos, user_neg, eligible
+
+
+def sample_bpr_pairs(user_pos, user_neg, eligible_users, n_pairs, rng):
+    """
+    Sample n_pairs (positive_idx, negative_idx) within-user pairs uniformly.
+    Returns arrays pos_idx, neg_idx of shape (n_pairs,).
+    """
+    pos_idx = np.empty(n_pairs, dtype=np.int64)
+    neg_idx = np.empty(n_pairs, dtype=np.int64)
+    n_eligible = len(eligible_users)
+
+    for i in range(n_pairs):
+        u = eligible_users[rng.integers(0, n_eligible)]
+        p = user_pos[u][rng.integers(0, len(user_pos[u]))]
+        n = user_neg[u][rng.integers(0, len(user_neg[u]))]
+        pos_idx[i] = p
+        neg_idx[i] = n
+
+    return pos_idx, neg_idx
+
+
+def bpr_step(model, X, pos_idx, neg_idx, lr=0.001, l2=1e-6):
+    """
+    One BPR update step.
+    Computes loss = -mean(log(sigmoid(z_pos - z_neg)))
+    and updates model weights via SGD-style gradient.
+    
+    We implement this by leveraging FM.logits() and manual gradient computation.
+    The FM score for a sample is: z = b + sum_f(W[x_f]) + sum_{f<g}(V[x_f] . V[x_g])
+    
+    For BPR, gradient wrt each parameter p:
+      dL/dp = -sigmoid(-diff) * (dz_pos/dp - dz_neg/dp)
+    where diff = z_pos - z_neg
+    """
+    X_pos = X[pos_idx]
+    X_neg = X[neg_idx]
+    B = len(pos_idx)
+    
+    # Get scores and intermediate values
+    z_pos, E_pos, S_pos = model.logits(X_pos)  # E: (B, F, k), S: (B, k)
+    z_neg, E_neg, S_neg = model.logits(X_neg)
+    
+    diff = z_pos - z_neg  # (B,)
+    # sigmoid(-diff) = 1 - sigmoid(diff)
+    sig_neg_diff = 1.0 / (1.0 + np.exp(diff))  # gradient multiplier: -sigmoid(-diff) = sig_neg_diff - 1... 
+    # Actually: d/dz_pos [-log(sigmoid(diff))] = -sigmoid(-diff) = -(1-sigmoid(diff)) = sigmoid(diff)-1
+    # So grad_multiplier for z_pos = sigmoid(diff) - 1 = -sigmoid(-diff)
+    # And grad_multiplier for z_neg = +sigmoid(-diff)
+    
+    # grad_multiplier: shape (B,)
+    # dL/dz_pos = (sigmoid(diff) - 1) / B = -sig_neg_diff / B
+    # dL/dz_neg = sig_neg_diff / B
+    
+    gm_pos = -sig_neg_diff / B  # (B,)
+    gm_neg = sig_neg_diff / B   # (B,)
+    
+    # Compute loss for monitoring
+    loss = -np.mean(np.log(np.maximum(1.0 - sig_neg_diff, 1e-15)))
+    
+    # ── Gradient for bias b ──────────────────────────────────────────────────
+    db = np.sum(gm_pos) + np.sum(gm_neg)
+    
+    # ── Gradient for W (first-order weights) ────────────────────────────────
+    # dz/dW[x_f] = 1 for each feature index x_f in the row
+    # X_pos: (B, F), each row has F feature indices
+    # We need to accumulate gradient: for each feature index, sum gm_pos over rows containing it
+    F = X_pos.shape[1]
+    k = model.V.shape[1]
+    
+    dW = np.zeros_like(model.W)  # shape: (dim,)
+    # Scatter add for positive items
+    for f in range(F):
+        np.add.at(dW, X_pos[:, f], gm_pos)
+        np.add.at(dW, X_neg[:, f], gm_neg)
+    
+    # ── Gradient for V (embeddings) ──────────────────────────────────────────
+    # FM interaction score: 0.5 * (||sum_f V[x_f]||^2 - sum_f ||V[x_f]||^2)
+    # dz/dV[x_f] = S - V[x_f]  where S = sum_g V[x_g]
+    # S_pos: (B, k) = sum of embeddings for each positive row
+    # E_pos: (B, F, k) = individual embeddings for each feature
+    
+    dV = np.zeros_like(model.V)  # shape: (dim, k)
+    
+    # For positive rows:
+    # dL/dV[x_f] += gm_pos[b] * (S_pos[b] - E_pos[b, f])
+    for f in range(F):
+        # grad_f = gm_pos[:, None] * (S_pos - E_pos[:, f, :])  shape: (B, k)
+        grad_f_pos = gm_pos[:, None] * (S_pos - E_pos[:, f, :])
+        np.add.at(dV, X_pos[:, f], grad_f_pos)
+        
+        grad_f_neg = gm_neg[:, None] * (S_neg - E_neg[:, f, :])
+        np.add.at(dV, X_neg[:, f], grad_f_neg)
+    
+    # ── L2 regularization ────────────────────────────────────────────────────
+    # Only regularize the features that appear in this batch
+    unique_pos = np.unique(X_pos)
+    unique_neg = np.unique(X_neg)
+    all_unique = np.unique(np.concatenate([unique_pos, unique_neg]))
+    
+    dW[all_unique] += l2 * model.W[all_unique]
+    dV[all_unique] += l2 * model.V[all_unique]
+    
+    # ── Adam update (use model's internal Adam state if available, else SGD) ──
+    # FM class uses Adam internally via step(), but we need manual control here.
+    # We'll use the model's Adam moment estimates directly.
+    # Check if model has Adam state; FM baseline stores t, m_V, v_V, m_W, v_W, m_b, v_b
+    
+    t = model.t + 1
+    model.t = t
+    beta1, beta2, eps = 0.9, 0.999, 1e-8
+    
+    # Update b
+    model.m_b = beta1 * model.m_b + (1 - beta1) * db
+    model.v_b = beta2 * model.v_b + (1 - beta2) * db * db
+    mb_hat = model.m_b / (1 - beta1 ** t)
+    vb_hat = model.v_b / (1 - beta2 ** t)
+    model.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
+    
+    # Update W
+    model.m_W += (1 - beta1) * (dW - model.m_W)  # This is wrong, let me fix
+    # Actually FM stores: m_W = beta1 * m_W + (1-beta1) * grad  -- standard Adam
+    # Let me redo this properly
+    model.m_W = beta1 * model.m_W + (1 - beta1) * dW
+    model.v_W = beta2 * model.v_W + (1 - beta2) * dW * dW
+    mW_hat = model.m_W / (1 - beta1 ** t)
+    vW_hat = model.v_W / (1 - beta2 ** t)
+    model.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
+    
+    # Update V
+    model.m_V = beta1 * model.m_V + (1 - beta1) * dV
+    model.v_V = beta2 * model.v_V + (1 - beta2) * dV * dV
+    mV_hat = model.m_V / (1 - beta1 ** t)
+    vV_hat = model.v_V / (1 - beta2 ** t)
+    model.V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
+    
+    return loss
+
+
+def run_bpr_epoch(model, X, users, labels, rng, batch_size=BATCH, lr=0.001, l2=1e-6):
+    """
+    One BPR epoch: sample pairs and update.
+    Number of batches = ceil(n_positives / batch_size) to match roughly one pass.
+    """
+    user_pos, user_neg, eligible = build_user_pos_neg_index(users, labels)
+    
+    if len(eligible) == 0:
+        return 0.0
+    
+    # Number of positive samples in training
+    n_pos = sum(len(v) for v in user_pos.values())
+    n_batches = max(1, n_pos // batch_size)
+    
     losses = []
-    for start in range(0, len(y), batch_size):
-        b = idx[start:start + batch_size]
-        loss = model.step(X[b], y[b])
+    for _ in range(n_batches):
+        pos_idx, neg_idx = sample_bpr_pairs(user_pos, user_neg, eligible, batch_size, rng)
+        loss = bpr_step(model, X, pos_idx, neg_idx, lr=lr, l2=l2)
         losses.append(loss)
+    
     return float(np.mean(losses))
 
 
-def train_single_model(seed, X_tr, y_tr, X_va, y_va, u_va, new_dim):
-    """Train one FM model with given seed, return best model and epoch count."""
+def train_single_model_bpr(seed, X_tr, y_tr, u_tr, X_va, y_va, u_va, new_dim):
+    """Train one FM model with BPR loss, return best model and epoch count."""
     rng = np.random.default_rng(seed)
     model = FM(new_dim, k=16, lr=0.001, l2=1e-6, seed=seed)
-
+    
+    # Build user index once
+    users_arr = np.array(u_tr)
+    labels_arr = y_tr
+    
     best_val = -1.0
     best_weights = None
     patience_left = PATIENCE
     best_epoch = 0
 
     for epoch in range(1, MAX_EPOCHS + 1):
-        tr_loss = run_epoch(model, X_tr, y_tr, rng)
+        tr_loss = run_bpr_epoch(model, X_tr, users_arr, labels_arr, rng,
+                                batch_size=BATCH, lr=0.001, l2=1e-6)
 
         val_scores = model.predict(X_va)
         val_res = evaluate(u_va, y_va, val_scores)
         val_primary = val_res['primary']
 
-        print(f"  Seed {seed} Epoch {epoch:02d}  tr_loss={tr_loss:.4f}  "
+        print(f"  Seed {seed} Epoch {epoch:02d}  bpr_loss={tr_loss:.4f}  "
               f"val_primary={val_primary:.4f}", flush=True)
 
         if val_primary > best_val + 1e-6:
@@ -235,7 +366,14 @@
             best_weights = (
                 model.V.copy(),
                 model.W.copy(),
-                float(model.b)
+                float(model.b),
+                model.m_V.copy(),
+                model.v_V.copy(),
+                model.m_W.copy(),
+                model.v_W.copy(),
+                float(model.m_b),
+                float(model.v_b),
+                int(model.t),
             )
             patience_left = PATIENCE
             best_epoch = epoch
@@ -259,7 +397,6 @@
     rand_rows = load_random_valid(args.data_dir)
     X_rand_base, y_rand, u_rand, _ = encode_like_train(splits['train'], rand_rows)
 
-    # Compute CTR features for random rows (use same maps from training)
     rand_video_rates = make_bucket_feature(rand_rows, video_ctr_map, video_global_rate, key_idx=2)
     rand_author_rates = make_bucket_feature(rand_rows, author_ctr_map, author_global_rate, key_idx=3)
     rand_video_buckets = compute_buckets_from_rates(train_video_rates, rand_video_rates, N_BINS)
@@ -276,16 +413,16 @@
 
     for s in range(N_SEEDS):
         print(f"\n=== Training seed {s} ===", flush=True)
-        model, best_epoch = train_single_model(s, X_tr, y_tr, X_va, y_va, u_va, NEW_DIM)
+        model, best_epoch = train_single_model_bpr(
+            s, X_tr, y_tr, u_tr, X_va, y_va, u_va, NEW_DIM
+        )
         best_epochs.append(best_epoch)
 
-        # Compute scores for each split
         va_scores = model.predict(X_va)
         te_scores = model.predict(X_te)
         tr_scores = model.predict(X_tr)
         rand_scores = model.predict(X_rand)
 
-        # Convert to within-user fractional ranks
         va_ranks = within_user_ranks(u_va, va_scores)
         te_ranks = within_user_ranks(u_te, te_scores)
         tr_ranks = within_user_ranks(u_tr, tr_scores)
@@ -296,7 +433,6 @@
         tr_rank_sum += tr_ranks
         rand_rank_sum += rand_ranks
 
-    # Average ranks
     va_final = va_rank_sum / N_SEEDS
     te_final = te_rank_sum / N_SEEDS
     tr_final = tr_rank_sum / N_SEEDS
@@ -304,11 +440,9 @@
 
     print(f"\nBest epochs across seeds: {best_epochs}", flush=True)
 
-    # ── Evaluate ──────────────────────────────────────────────────────────────
     tr_res = evaluate(u_tr, y_tr, tr_final)
     va_res = evaluate(u_va, y_va, va_final)
 
-    # Unbiased evaluation using averaged ranks
     unbiased = unbiased_primary(
         args.data_dir, splits['train'],
         lambda rows: rand_final
@@ -320,7 +454,6 @@
     print(f"VAL_PRIMARY={va_res['primary']:.4f}")
     print(f"UNBIASED_PRIMARY={unbiased:.4f}")
 
-    # ── Write submissions ──────────────────────────────────────────────────────
     submit.write_submission(
         os.path.join(args.out_dir, 'submission_valid.csv'),
         splits['valid'], va_final
@@ -331,16 +464,12 @@
     )
 
 else:
-    # train+valid mode: combine train and valid, train for fixed epoch count
-    # Need to recompute CTR features combining train+valid for proper features
-    # BUT: we must not recompute CTR maps from train+valid (would be leakage for test scoring)
-    # CTR maps remain from training only; just extend features for valid rows as done above
-
+    # train+valid mode
     X_all = np.concatenate([X_tr, X_va], axis=0)
     y_all = np.concatenate([y_tr, y_va], axis=0)
     u_all = list(u_tr) + list(u_va)
 
-    FIXED_EPOCHS = 10  # conservative fixed schedule for train+valid
+    FIXED_EPOCHS = 7  # median of best epochs from train run
 
     te_rank_sum = np.zeros(len(y_te), dtype=np.float64)
 
@@ -348,10 +477,12 @@
         print(f"\n=== Training seed {s} (train+valid) ===", flush=True)
         rng = np.random.default_rng(s)
         model = FM(NEW_DIM, k=16, lr=0.001, l2=1e-6, seed=s)
+        users_all_arr = np.array(u_all)
 
         for epoch in range(1, FIXED_EPOCHS + 1):
-            tr_loss = run_epoch(model, X_all, y_all, rng)
-            print(f"  Seed {s} Epoch {epoch:02d}  tr_loss={tr_loss:.4f}", flush=True)
+            tr_loss = run_bpr_epoch(model, X_all, users_all_arr, y_all, rng,
+                                    batch_size=BATCH, lr=0.001, l2=1e-6)
+            print(f"  Seed {s} Epoch {epoch:02d}  bpr_loss={tr_loss:.4f}", flush=True)
 
         te_scores = model.predict(X_te)
         te_ranks = within_user_ranks(u_te, te_scores)
```

## Node 5 — `improve` (parent: node 3)

**Error:** `llm_failure` (LLMError)
**Recovery:** fall back to the best node next iteration

_No code file on disk for this node (see error above)._

## Node 6 — `improve` (parent: node 3)

**Hypothesis:**

Replacing pointwise BCE with BPR pairwise loss — sampling within-user (positive, negative) pairs and training on sigmoid(score_pos - score_neg) — will improve GAUC and thus validation primary by optimizing the same objective the metric measures.

**Metrics:**

| train_primary | val_gauc | val_ndcg5 | val_primary | unbiased | diagnosis |
|---|---|---|---|---|---|
| 0.6440 | 0.6711 | 0.5373 | **0.6042** | 0.3672 | noise |

**Accepted:** yes — became the new best
**Leakage review:** CLEAN

**Code diff** (+285/-87 lines vs. node 3):

```diff
--- node_3/solution.py
+++ node_6/solution.py
@@ -1,12 +1,11 @@
 #!/usr/bin/env python3
 """
-Iteration 2: Add Bayesian-smoothed per-video and per-author long_view rates (bucketed)
-to the 5-seed FM ensemble with within-user rank averaging.
-
-Changes from iteration 1:
-- Compute Bayesian-smoothed long_view rate for each video and author from training data only
-- Bucket each into 10 equal-frequency bins
-- Add these 2 new categorical features to FM input (7 features total instead of 5)
+Iteration 4: Replace pointwise BCE with BPR pairwise loss.
+
+Changes from iteration 2/3:
+- BPR pairwise training: sample (pos, neg) pairs within each minibatch per user
+- Optimize sigmoid(score_pos - score_neg) directly, which aligns with GAUC objective
+- Keep: 7 features, 5-seed ensemble, rank averaging, Bayesian-smoothed CTR buckets
 """
 
 import argparse
@@ -27,7 +26,6 @@
 
 # ── Imports ───────────────────────────────────────────────────────────────────
 from data import load, encode, FIELDS
-from baseline import FM
 from evaluate import evaluate
 import submit
 from unbiased import load_random_valid, encode_like_train, unbiased_primary
@@ -41,19 +39,10 @@
 X_te_base, y_te, u_te = enc['test']
 
 # ── Compute Bayesian-smoothed CTR features from training data only ────────────
-# Fields: user_id=0, video_id=1, author_id=2, tab=3, dur_bucket=4, long_view=5 in raw rows
-# We use splits['train'] which has tuples: (date, user_id, video_id, author_id, tab, duration_ms, long_view)
-
-GLOBAL_MEAN_PRIOR_STRENGTH = 50  # pseudo-count for Bayesian smoothing
+GLOBAL_MEAN_PRIOR_STRENGTH = 50
 
 def compute_bayesian_ctr(train_rows, key_idx, label_idx=6):
-    """
-    Compute Bayesian-smoothed long_view rate keyed by field at key_idx.
-    Returns a dict: key -> smoothed_rate
-    key_idx: index into the row tuple (1=user_id, 2=video_id, 3=author_id)
-    label_idx: index of long_view label (6 in load() tuples)
-    """
-    counts = {}  # key -> [total, positives]
+    counts = {}
     for row in train_rows:
         key = row[key_idx]
         label = row[label_idx]
@@ -62,12 +51,10 @@
         counts[key][0] += 1
         counts[key][1] += label
 
-    # Global rate (prior)
     total_all = sum(v[0] for v in counts.values())
     pos_all = sum(v[1] for v in counts.values())
     global_rate = pos_all / total_all if total_all > 0 else 0.5
 
-    # Bayesian smoothing: rate = (positives + alpha * global_rate) / (total + alpha)
     alpha = GLOBAL_MEAN_PRIOR_STRENGTH
     smoothed = {}
     for key, (total, pos) in counts.items():
@@ -76,12 +63,7 @@
     return smoothed, global_rate
 
 
-def make_bucket_feature(rows, ctr_map, global_rate, key_idx, n_bins=10):
-    """
-    For each row, look up the Bayesian-smoothed CTR and bucket it.
-    Returns integer array of bucket indices (0 to n_bins-1).
-    Unknown keys get global_rate.
-    """
+def make_bucket_feature(rows, ctr_map, global_rate, key_idx):
     rates = np.array([
         ctr_map.get(row[key_idx], global_rate)
         for row in rows
@@ -90,37 +72,26 @@
 
 
 def compute_buckets_from_rates(train_rates, query_rates, n_bins=10):
-    """
-    Use quantile edges from training rates to bucket query rates.
-    Returns integer bucket indices (0 to n_bins-1).
-    """
-    # Compute quantile edges from training rates
     quantiles = np.linspace(0, 100, n_bins + 1)
     edges = np.percentile(train_rates, quantiles)
-    # Ensure edges are unique; if not, use unique edges
     edges = np.unique(edges)
-    # Bucket query rates using these edges
-    # np.searchsorted: values below first edge -> 0, above last -> len(edges)
     buckets = np.searchsorted(edges[1:-1], query_rates, side='right')
-    # Clip to [0, n_bins-1]
     buckets = np.clip(buckets, 0, n_bins - 1)
     return buckets.astype(np.int32)
 
 
-# Compute CTR maps from training data only
 train_rows = splits['train']
 valid_rows = splits['valid']
 test_rows = splits['test']
 
 print("Computing video CTR map...", flush=True)
-video_ctr_map, video_global_rate = compute_bayesian_ctr(train_rows, key_idx=2)  # video_id at idx 2
+video_ctr_map, video_global_rate = compute_bayesian_ctr(train_rows, key_idx=2)
 print(f"  {len(video_ctr_map)} videos, global rate={video_global_rate:.4f}", flush=True)
 
 print("Computing author CTR map...", flush=True)
-author_ctr_map, author_global_rate = compute_bayesian_ctr(train_rows, key_idx=3)  # author_id at idx 3
+author_ctr_map, author_global_rate = compute_bayesian_ctr(train_rows, key_idx=3)
 print(f"  {len(author_ctr_map)} authors, global rate={author_global_rate:.4f}", flush=True)
 
-# Get rates for all splits
 N_BINS = 10
 
 train_video_rates = make_bucket_feature(train_rows, video_ctr_map, video_global_rate, key_idx=2)
@@ -132,7 +103,6 @@
 test_video_rates = make_bucket_feature(test_rows, video_ctr_map, video_global_rate, key_idx=2)
 test_author_rates = make_bucket_feature(test_rows, author_ctr_map, author_global_rate, key_idx=3)
 
-# Compute bucket assignments using training quantiles
 train_video_buckets = compute_buckets_from_rates(train_video_rates, train_video_rates, N_BINS)
 train_author_buckets = compute_buckets_from_rates(train_author_rates, train_author_rates, N_BINS)
 
@@ -141,15 +111,6 @@
 
 test_video_buckets = compute_buckets_from_rates(train_video_rates, test_video_rates, N_BINS)
 test_author_buckets = compute_buckets_from_rates(train_author_rates, test_author_rates, N_BINS)
-
-print(f"Video CTR bucket distribution (train): {np.bincount(train_video_buckets)}", flush=True)
-print(f"Author CTR bucket distribution (train): {np.bincount(train_author_buckets)}", flush=True)
-
-# ── Extend feature matrices with new CTR bucket features ─────────────────────
-# Add offset to new feature indices to avoid collision with existing features
-# Existing dim covers indices 0..dim-1
-# New features: video_ctr_bucket (0..N_BINS-1) and author_ctr_bucket (0..N_BINS-1)
-# They each get their own slot range after dim
 
 VIDEO_CTR_OFFSET = dim
 AUTHOR_CTR_OFFSET = dim + N_BINS
@@ -157,17 +118,238 @@
 
 print(f"Original embedding dim: {dim}, New dim: {NEW_DIM}", flush=True)
 
+
 def extend_features(X_base, video_buckets, author_buckets):
-    """Append video_ctr and author_ctr bucket columns to feature matrix."""
     vid_col = (video_buckets + VIDEO_CTR_OFFSET).reshape(-1, 1).astype(np.int32)
     auth_col = (author_buckets + AUTHOR_CTR_OFFSET).reshape(-1, 1).astype(np.int32)
     return np.concatenate([X_base, vid_col, auth_col], axis=1)
 
+
 X_tr = extend_features(X_tr_base, train_video_buckets, train_author_buckets)
 X_va = extend_features(X_va_base, valid_video_buckets, valid_author_buckets)
 X_te = extend_features(X_te_base, test_video_buckets, test_author_buckets)
 
 print(f"Feature matrix shapes: train={X_tr.shape}, valid={X_va.shape}, test={X_te.shape}", flush=True)
+
+# ── FM model with BPR training ────────────────────────────────────────────────
+# We implement our own FM with Adam optimizer to support BPR loss
+# FM score: b + sum_f(W[f]*x[f]) + sum_{f<g}(V[f]·V[g])
+# Using the kernel trick: interaction = 0.5*(||sum_f V[f]||^2 - sum_f ||V[f]||^2)
+
+class FMModel:
+    """
+    Factorization Machine with Adam optimizer.
+    Supports both pointwise BCE (for warm-up) and BPR pairwise training.
+    """
+    def __init__(self, n_features, k=16, lr=0.001, l2=1e-6, seed=0):
+        rng = np.random.default_rng(seed)
+        self.k = k
+        self.lr = lr
+        self.l2 = l2
+        self.n_features = n_features
+
+        # Parameters
+        self.V = rng.normal(0, 0.01, (n_features, k)).astype(np.float32)
+        self.W = np.zeros(n_features, dtype=np.float32)
+        self.b = np.float32(0.0)
+
+        # Adam state for V
+        self.m_V = np.zeros_like(self.V)
+        self.v_V = np.zeros_like(self.V)
+        self.m_W = np.zeros_like(self.W)
+        self.v_W = np.zeros_like(self.W)
+        self.m_b = 0.0
+        self.v_b = 0.0
+
+        self.t = 0  # Adam step count
+        self.beta1 = 0.9
+        self.beta2 = 0.999
+        self.eps = 1e-8
+
+    def _fm_score(self, X):
+        """
+        Compute FM scores for rows in X.
+        X: int32 array (B, F) of feature indices
+        Returns: (scores, sum_emb, sq_sum_emb) for gradient computation
+        """
+        B, F = X.shape
+        # First order
+        scores = np.full(B, self.b, dtype=np.float64)
+        for f in range(F):
+            idx = X[:, f]
+            scores += self.W[idx]
+
+        # Second order interaction via kernel trick
+        # sum_emb[b] = sum_f V[X[b,f]] shape (B, k)
+        sum_emb = np.zeros((B, self.k), dtype=np.float64)
+        sq_sum_emb = np.zeros((B, self.k), dtype=np.float64)
+        for f in range(F):
+            idx = X[:, f]
+            vf = self.V[idx].astype(np.float64)
+            sum_emb += vf
+            sq_sum_emb += vf ** 2
+
+        # interaction = 0.5 * (||sum_emb||^2 - sum(sq_sum_emb))
+        interaction = 0.5 * (np.sum(sum_emb ** 2, axis=1) - np.sum(sq_sum_emb, axis=1))
+        scores += interaction
+        return scores.astype(np.float32), sum_emb.astype(np.float32), sq_sum_emb.astype(np.float32)
+
+    def _adam_update(self, param, grad, m, v):
+        """One Adam step. Returns updated param, m, v."""
+        m = self.beta1 * m + (1 - self.beta1) * grad
+        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
+        m_hat = m / (1 - self.beta1 ** self.t)
+        v_hat = v / (1 - self.beta2 ** self.t)
+        param = param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
+        return param, m, v
+
+    def predict(self, X, bs=200_000):
+        """Predict scores for all rows in X."""
+        scores = []
+        for start in range(0, len(X), bs):
+            Xb = X[start:start + bs]
+            s, _, _ = self._fm_score(Xb)
+            scores.append(s)
+        return np.concatenate(scores)
+
+    def bpr_step(self, X_pos, X_neg):
+        """
+        One BPR update step on a batch of (positive, negative) pairs.
+        X_pos, X_neg: int32 arrays (P, F) where P = number of pairs
+        Returns: mean BPR loss
+        """
+        if len(X_pos) == 0:
+            return 0.0
+
+        self.t += 1
+        P, F = X_pos.shape
+
+        # Compute scores for positive and negative items
+        s_pos, sum_pos, sq_pos = self._fm_score(X_pos)
+        s_neg, sum_neg, sq_neg = self._fm_score(X_neg)
+
+        # BPR: loss = -mean(log(sigmoid(s_pos - s_neg)))
+        diff = s_pos.astype(np.float64) - s_neg.astype(np.float64)
+        # sigmoid(diff)
+        sig = 1.0 / (1.0 + np.exp(-np.clip(diff, -30, 30)))
+
+        loss = -np.mean(np.log(sig + 1e-10))
+
+        # Gradient: d_loss/d_diff = -(1 - sigmoid(diff)) per pair
+        # d_loss/d_s_pos = -(1-sig)/P, d_loss/d_s_neg = +(1-sig)/P
+        grad_diff = -(1.0 - sig) / P  # shape (P,)
+
+        # Now backprop through FM for pos and neg
+        # For each row i, d_loss/d_b += grad_diff[i]
+        # d_loss/d_W[idx] += grad_diff[i] * 1  (first-order)
+        # d_loss/d_V[idx,j] += grad_diff[i] * (sum_emb[i,j] - V[idx,j])  (interaction)
+
+        # --- Bias gradient ---
+        db = np.sum(grad_diff) - np.sum(grad_diff)  # cancel: pos and neg each contribute
+        # Actually: grad for s_pos goes to pos rows, grad for s_neg goes to neg rows
+        db_pos = np.sum(grad_diff)
+        db_neg = -np.sum(grad_diff)
+        db = db_pos + db_neg  # = 0, but let's keep it general
+
+        # --- W gradient ---
+        dW = np.zeros(self.n_features, dtype=np.float64)
+        for f in range(F):
+            idx_pos = X_pos[:, f]
+            idx_neg = X_neg[:, f]
+            np.add.at(dW, idx_pos, grad_diff)
+            np.add.at(dW, idx_neg, -grad_diff)
+
+        # L2 regularization on W
+        dW += self.l2 * self.W
+
+        # --- V gradient ---
+        dV = np.zeros((self.n_features, self.k), dtype=np.float64)
+
+        sum_pos_d = sum_pos.astype(np.float64)
+        sum_neg_d = sum_neg.astype(np.float64)
+        V_d = self.V.astype(np.float64)
+
+        for f in range(F):
+            idx_pos = X_pos[:, f]
+            idx_neg = X_neg[:, f]
+
+            # For positive rows: grad = grad_diff[:, None] * (sum_pos - V[idx_pos])
+            v_pos_f = V_d[idx_pos]  # (P, k)
+            g_pos = grad_diff[:, None] * (sum_pos_d - v_pos_f)  # (P, k)
+            np.add.at(dV, idx_pos, g_pos)
+
+            # For negative rows: grad = -grad_diff[:, None] * (sum_neg - V[idx_neg])
+            v_neg_f = V_d[idx_neg]  # (P, k)
+            g_neg = -grad_diff[:, None] * (sum_neg_d - v_neg_f)  # (P, k)
+            np.add.at(dV, idx_neg, g_neg)
+
+        # L2 regularization on V (only for indices that appear)
+        touched = np.unique(np.concatenate([X_pos.ravel(), X_neg.ravel()]))
+        dV[touched] += self.l2 * V_d[touched]
+
+        # --- Adam updates ---
+        self.b, self.m_b, self.v_b = self._adam_update(
+            float(self.b), db, self.m_b, self.v_b
+        )
+        self.b = np.float32(self.b)
+
+        self.W, self.m_W, self.v_W = self._adam_update(
+            self.W.astype(np.float64), dW, self.m_W, self.v_W
+        )
+        self.W = self.W.astype(np.float32)
+
+        self.V, self.m_V, self.v_V = self._adam_update(
+            self.V.astype(np.float64), dV, self.m_V, self.v_V
+        )
+        self.V = self.V.astype(np.float32)
+
+        return float(loss)
+
+
+def build_bpr_pairs(X, y, users, rng):
+    """
+    For each user in batch that has at least one positive and one negative,
+    sample one (positive, negative) pair.
+    
+    Returns: X_pos (P, F), X_neg (P, F) arrays
+    """
+    y_arr = np.asarray(y)
+    users_arr = np.asarray(users)
+
+    # Group by user
+    user_to_pos = {}
+    user_to_neg = {}
+
+    for i in range(len(y_arr)):
+        u = users_arr[i]
+        if y_arr[i] == 1:
+            if u not in user_to_pos:
+                user_to_pos[u] = []
+            user_to_pos[u].append(i)
+        else:
+            if u not in user_to_neg:
+                user_to_neg[u] = []
+            user_to_neg[u].append(i)
+
+    pos_indices = []
+    neg_indices = []
+
+    for u in user_to_pos:
+        if u in user_to_neg:
+            # Sample one positive and one negative
+            p_idx = rng.choice(user_to_pos[u])
+            n_idx = rng.choice(user_to_neg[u])
+            pos_indices.append(p_idx)
+            neg_indices.append(n_idx)
+
+    if len(pos_indices) == 0:
+        return np.zeros((0, X.shape[1]), dtype=np.int32), np.zeros((0, X.shape[1]), dtype=np.int32)
+
+    pos_indices = np.array(pos_indices)
+    neg_indices = np.array(neg_indices)
+
+    return X[pos_indices], X[neg_indices]
+
 
 # ── Hyperparameters ───────────────────────────────────────────────────────────
 BATCH = 8192
@@ -177,10 +359,6 @@
 
 # ── Helper: within-user fractional ranks ──────────────────────────────────────
 def within_user_ranks(users, scores):
-    """
-    For each user, convert raw scores to fractional ranks in [0, 1].
-    Rank 0 = lowest score, rank 1 = highest score (within that user's items).
-    """
     user_arr = np.array(users)
     ranks = np.zeros(len(scores), dtype=np.float64)
     unique_users = np.unique(user_arr)
@@ -198,22 +376,39 @@
         ranks[mask] = r
     return ranks
 
-# ── Training helper ───────────────────────────────────────────────────────────
-def run_epoch(model, X, y, rng, batch_size=BATCH):
-    """One full pass over data in random order, returns mean loss."""
+
+def run_bpr_epoch(model, X, y, users, rng, batch_size=BATCH):
+    """One full pass over data with BPR pairwise loss."""
     idx = rng.permutation(len(y))
+    X_shuf = X[idx]
+    y_shuf = y[idx]
+    users_shuf = [users[i] for i in idx]
+
     losses = []
+    n_pairs_total = 0
+
     for start in range(0, len(y), batch_size):
-        b = idx[start:start + batch_size]
-        loss = model.step(X[b], y[b])
-        losses.append(loss)
-    return float(np.mean(losses))
-
-
-def train_single_model(seed, X_tr, y_tr, X_va, y_va, u_va, new_dim):
-    """Train one FM model with given seed, return best model and epoch count."""
+        end = min(start + batch_size, len(y))
+        Xb = X_shuf[start:end]
+        yb = y_shuf[start:end]
+        ub = users_shuf[start:end]
+
+        X_pos, X_neg = build_bpr_pairs(Xb, yb, ub, rng)
+        n_pairs = len(X_pos)
+        n_pairs_total += n_pairs
+
+        if n_pairs > 0:
+            loss = model.bpr_step(X_pos, X_neg)
+            losses.append(loss)
+
+    avg_loss = float(np.mean(losses)) if losses else 0.0
+    return avg_loss, n_pairs_total
+
+
+def train_single_model_bpr(seed, X_tr, y_tr, u_tr, X_va, y_va, u_va, new_dim):
+    """Train one FM model with BPR loss, return best model and epoch count."""
     rng = np.random.default_rng(seed)
-    model = FM(new_dim, k=16, lr=0.001, l2=1e-6, seed=seed)
+    model = FMModel(new_dim, k=16, lr=0.001, l2=1e-6, seed=seed)
 
     best_val = -1.0
     best_weights = None
@@ -221,21 +416,26 @@
     best_epoch = 0
 
     for epoch in range(1, MAX_EPOCHS + 1):
-        tr_loss = run_epoch(model, X_tr, y_tr, rng)
+        tr_loss, n_pairs = run_bpr_epoch(model, X_tr, y_tr, u_tr, rng)
 
         val_scores = model.predict(X_va)
         val_res = evaluate(u_va, y_va, val_scores)
         val_primary = val_res['primary']
 
-        print(f"  Seed {seed} Epoch {epoch:02d}  tr_loss={tr_loss:.4f}  "
-              f"val_primary={val_primary:.4f}", flush=True)
+        print(f"  Seed {seed} Epoch {epoch:02d}  bpr_loss={tr_loss:.4f}  "
+              f"n_pairs={n_pairs}  val_primary={val_primary:.4f}", flush=True)
 
         if val_primary > best_val + 1e-6:
             best_val = val_primary
             best_weights = (
                 model.V.copy(),
                 model.W.copy(),
-                float(model.b)
+                float(model.b),
+                # Save Adam state too
+                model.m_V.copy(), model.v_V.copy(),
+                model.m_W.copy(), model.v_W.copy(),
+                model.m_b, model.v_b,
+                model.t
             )
             patience_left = PATIENCE
             best_epoch = epoch
@@ -247,9 +447,10 @@
                 break
 
     # Restore best weights
-    model.V[:] = best_weights[0]
-    model.W[:] = best_weights[1]
-    model.b = best_weights[2]
+    if best_weights is not None:
+        model.V[:] = best_weights[0]
+        model.W[:] = best_weights[1]
+        model.b = np.float32(best_weights[2])
 
     return model, best_epoch
 
@@ -259,7 +460,6 @@
     rand_rows = load_random_valid(args.data_dir)
     X_rand_base, y_rand, u_rand, _ = encode_like_train(splits['train'], rand_rows)
 
-    # Compute CTR features for random rows (use same maps from training)
     rand_video_rates = make_bucket_feature(rand_rows, video_ctr_map, video_global_rate, key_idx=2)
     rand_author_rates = make_bucket_feature(rand_rows, author_ctr_map, author_global_rate, key_idx=3)
     rand_video_buckets = compute_buckets_from_rates(train_video_rates, rand_video_rates, N_BINS)
@@ -275,8 +475,10 @@
     best_epochs = []
 
     for s in range(N_SEEDS):
-        print(f"\n=== Training seed {s} ===", flush=True)
-        model, best_epoch = train_single_model(s, X_tr, y_tr, X_va, y_va, u_va, NEW_DIM)
+        print(f"\n=== Training seed {s} (BPR) ===", flush=True)
+        model, best_epoch = train_single_model_bpr(
+            s, X_tr, y_tr, u_tr, X_va, y_va, u_va, NEW_DIM
+        )
         best_epochs.append(best_epoch)
 
         # Compute scores for each split
@@ -331,27 +533,23 @@
     )
 
 else:
-    # train+valid mode: combine train and valid, train for fixed epoch count
-    # Need to recompute CTR features combining train+valid for proper features
-    # BUT: we must not recompute CTR maps from train+valid (would be leakage for test scoring)
-    # CTR maps remain from training only; just extend features for valid rows as done above
-
+    # train+valid mode
     X_all = np.concatenate([X_tr, X_va], axis=0)
     y_all = np.concatenate([y_tr, y_va], axis=0)
     u_all = list(u_tr) + list(u_va)
 
-    FIXED_EPOCHS = 10  # conservative fixed schedule for train+valid
+    FIXED_EPOCHS = 10
 
     te_rank_sum = np.zeros(len(y_te), dtype=np.float64)
 
     for s in range(N_SEEDS):
-        print(f"\n=== Training seed {s} (train+valid) ===", flush=True)
+        print(f"\n=== Training seed {s} (train+valid, BPR) ===", flush=True)
         rng = np.random.default_rng(s)
-        model = FM(NEW_DIM, k=16, lr=0.001, l2=1e-6, seed=s)
+        model = FMModel(NEW_DIM, k=16, lr=0.001, l2=1e-6, seed=s)
 
         for epoch in range(1, FIXED_EPOCHS + 1):
-            tr_loss = run_epoch(model, X_all, y_all, rng)
-            print(f"  Seed {s} Epoch {epoch:02d}  tr_loss={tr_loss:.4f}", flush=True)
+            tr_loss, n_pairs = run_bpr_epoch(model, X_all, y_all, u_all, rng)
+            print(f"  Seed {s} Epoch {epoch:02d}  bpr_loss={tr_loss:.4f}  n_pairs={n_pairs}", flush=True)
 
         te_scores = model.predict(X_te)
         te_ranks = within_user_ranks(u_te, te_scores)
```

## Node 7 — `draft`

**Hypothesis:**

The best result so far (0.6042) uses BPR pairwise loss with an FM model plus Bayesian-smoothed CTR features and seed ensembling. The next untested mechanism is **LightGBM with gradient-boosted trees** as the scoring model instead of FM. GBDT can capture non-linear interactions between features natively without discretizing continuous inputs, and with the right feature set (user history statistics crossed with video attributes) it may handle the within-user ranking task better. The key insight: GBDT accepts continuous features directly (duration_ms, smoothed CTR rates), can capture threshold effects (e.g., "if duration > X and user typically watches long videos"), and with LambdaRank/LambdaMART objective can optimize ranking directly. Since we're ranking within users, I'll use LightGBM's `lambdarank` objective with query groups defined by user_id. Features: the 5 standard fields encoded + continuous features (duration_ms, video CTR, author CTR computed from training history). Expected gain: +0.003 to +0.010 over baseline if ranking objective + continuous features help.

**Metrics:**

| train_primary | val_gauc | val_ndcg5 | val_primary | unbiased | diagnosis |
|---|---|---|---|---|---|
| 0.6315 | 0.6617 | 0.5338 | **0.5978** | 0.3633 | regression |

**Accepted:** no
**Leakage review:** CLEAN

**Code diff:** none — fresh file, no parent (358 lines). See `nodes/node_7/solution.py`.

