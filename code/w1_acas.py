#!/usr/bin/env python3
"""
W1-ACAS: 1-Wasserstein Adaptive Conformal Anomaly Score.

Faithful reimplementation of Algorithm 1 in Martinez Gil, O'Donncha, Gifford,
Zhou, Patel, Vaculin, "Adaptive Conformal Anomaly Detection with Time Series
Foundation Models for Signal Monitoring", ICLR 2026 (arXiv:2604.20122),
Sections 4-5 (Eqs. 6-14, Algorithm 1 box).

This module implements ONLY the general conformal-anomaly-score machinery
(Sec. 3-5): given an arbitrary sequence of nonconformity scores {S_t}, it
produces a sequence of normalized, adaptively-weighted p-values {beta_t} in
[0,1] such that flagging beta_t < alpha_c gives (asymptotically) an alpha_c
false-alarm rate under distribution shift. It does NOT reimplement the
paper's Sec. 6 TSFM forecaster (that machinery only supplies S_t = |Y - Yhat|
for their own benchmarks); here S_t is supplied externally and can be any
nonconformity score, including a frozen detector's smoothed anomaly score.
This substitution is legitimate because Sec. 3-5's derivation and Prop. 4.1
guarantee hold for ANY exchangeable-under-the-null sequence of scores, not
specifically forecast residuals (the paper states this explicitly at the
opening of Sec. 4).

Known deviations from the paper (state explicitly, do not silently omit):
  - We do not reimplement the multi-horizon aggregation (paper Sec. 4, "median
    across horizons"): our S_t already has a single horizon (one score per
    window), so D=1 and no aggregation is needed.
  - Hyperparameters (n, n_b, gamma, alpha_c) are not published as a table in
    the excerpt we could extract; we set them to values consistent with the
    paper's stated constraints (n_c = 1/alpha_c - 1) and our own corridor
    window counts, and report the exact values used.
"""
import numpy as np


def w1_acas(scores, alpha_c=0.01, n=400, n_b=20, gamma=0.05, w_init=None):
    """
    Algorithm 1 (W1-ACAS), applied verbatim to an arbitrary nonconformity-score
    sequence `scores` (1-D array, length T).

    Parameters
    ----------
    scores : array-like, shape (T,)
        The nonconformity score sequence S_1..S_T (here: the frozen detector's
        smoothed anomaly score s_t, treated as a generic nonconformity score
        per the paper's Sec. 3.1 framing).
    alpha_c : float
        Critical (minimum) false-alarm rate the method is calibrated to honor
        (paper's alpha_c). Determines the initial "critical sample size"
        n_c = 1/alpha_c - 1 (paper Algorithm 1, first line).
    n : int
        Maximum number of past scores retained in the sliding window used to
        compute each beta_t (paper's "n max past samples").
    n_b : int
        Batch size for the periodic weight update (paper's "nb min batch
        size"); weights are updated every n_b new observations.
    gamma : float
        Projected-gradient step size (paper's gamma in Eq. 12).
    w_init : array-like or None
        Optional explicit initial weight vector of length n; defaults to the
        paper's stated initialization w_i = 1[i <= n_c] for i in 1..n.

    Returns
    -------
    beta : np.ndarray, shape (T,), dtype float, with NaN for the first n_c
        indices (paper's algorithm starts at t = n_c, i.e. there is a burn-in
        of n_c samples before the first p-value can be produced).
    diagnostics : dict with the realized n_c, and the final weight vector.
    """
    scores = np.asarray(scores, dtype=float)
    T = len(scores)
    n_c = int(np.ceil(1.0 / alpha_c - 1.0))
    if n_c < 1:
        n_c = 1
    if n_c >= T:
        raise ValueError(
            f"n_c={n_c} (from alpha_c={alpha_c}) >= sequence length T={T}; "
            "cannot burn in. Use a larger alpha_c or a longer sequence."
        )

    # w indexed 1..n in the paper (1-based); we keep 0-based arrays of length n
    # and translate index arithmetic explicitly at each use site to avoid
    # silent off-by-one errors.
    if w_init is None:
        w = np.array([1.0 if (i + 1) <= n_c else 0.0 for i in range(n)])
    else:
        w = np.asarray(w_init, dtype=float).copy()
        assert len(w) == n

    beta = np.full(T, np.nan)
    # Rolling buffers for the batch update: the last n_b beta_{t+i} values and
    # the corresponding Jacobian rows d(beta_{t+i})/d(w_k), k=1..n.
    batch_beta = []
    batch_jac = []   # each entry: np.ndarray shape (n,) — d beta / d w_k for that step

    def project(w_vec):
        """Project onto {w in [0,1]^n : sum(w) > n_c}. Clip to [0,1] first;
        if the sum constraint is violated, rescale up toward the unit box
        (paper does not spell out the exact projection operator beyond the
        product-of-box-and-sum-floor set; we use the simplest feasible
        operator: clip, then if infeasible, uniformly lift all *active*
        (>0 after clipping) coordinates by the minimal amount needed, capped
        at 1.0, which stays inside [0,1]^n by construction)."""
        w_c = np.clip(w_vec, 0.0, 1.0)
        s = w_c.sum()
        target = n_c + 1e-6
        if s > target:
            return w_c
        # Need to raise mass. Lift uniformly on coordinates not already at 1.
        headroom = (1.0 - w_c)
        total_headroom = headroom.sum()
        if total_headroom <= 1e-12:
            # Already saturated at all-ones; nothing more to give.
            return w_c
        deficit = target - s
        frac = min(1.0, deficit / total_headroom)
        return w_c + frac * headroom

    w = project(w)

    for t in range(n_c, T):  # t is 0-based index of S_{t+1} in the paper's 1-based t+1 convention
        # s = past scores in window [max(t-n, 0), t) i.e. up to and excluding
        # the current test point, matching the paper's s = {S_i}_{i=max(t-n,1)}^{t}
        # (paper's t there is our t, 1-based; we use 0-based t meaning "index
        # of the sample being tested is t", so its strict past is [0..t-1]).
        lo = max(t - n, 0)
        s_win = scores[lo:t]          # chronological order, oldest..newest
        m = len(s_win)
        if m == 0:
            continue

        # w-hat_i = w_{|s|+1-i} for i=1..|s| (paper Eq. under Algorithm 1 box):
        # index 1 of w-hat <-> index |s| of w (i.e. reverse order), using the
        # FIRST m entries of w (w is indexed by recency-from-now, most-recent
        # first, per its initialization w_i=1[i<=n_c] meaning the n_c most
        # recent scores are trusted at burn-in).
        w_head = w[:m]                 # w_1..w_m (most-recent-first)
        w_hat = w_head[::-1]           # reverse -> matches s_win's chronological order

        s_test = scores[t]
        # j_{t+1} = #{past scores STRICTLY GREATER than the test score}
        j = int(np.sum(s_win > s_test))

        order = np.argsort(s_win, kind="mergesort")     # ascending; order[k] = index (in s_win) of k-th smallest
        rank_of = np.empty(m, dtype=int)                 # rank_of[i] = rank (0-based, ascending) of s_win[i]
        rank_of[order] = np.arange(m)

        # beta_{t+1} = (1 + sum_{k=j+1}^{m} w_hat[pi^{-1}(k)]) / (||w_hat||_1 + 1)
        # pi^{-1}(k) (1-based k) = order[k-1] (0-based). Sum over the top
        # (m - j) ranks (i.e. scores >= the test score's rank threshold j).
        if j < m:
            top_idx = order[j:]              # indices (into s_win/w_hat) of the (m-j) largest past scores
            num = 1.0 + w_hat[top_idx].sum()
        else:
            num = 1.0
        den = w_hat.sum() + 1.0
        b = num / den
        beta[t] = b

        # d beta_{t+1} / d w_k  (paper Eq. 14), for k=1..m, expressed here over
        # the ORIGINAL w-index space (0-based positions 0..m-1 of `w`, i.e.
        # w_1..w_m in the paper's notation, matching w_head above).
        # Eq. 14: d beta_{t+i}/d w_k = (-beta_{t+i}(w) + 1[j_{t+i} <= pi(k)]) / (||w||_1 + 1)
        # pi(k): rank (1-based) of the k-th ENTRY OF s (not of w-hat); since
        # w_hat[i] corresponds to s_win[m-1-i] (reverse map), we compute the
        # derivative in s_win-index space then map back to w-index space.
        pi_1based = rank_of + 1     # pi(k) for k = index into s_win (0-based k -> 1-based pi)
        indicator = (j <= pi_1based).astype(float)
        d_beta_d_s_index = (-b + indicator) / den   # length m, indexed by s_win position
        # map back: w_hat[i] = w[m-1-i]  =>  s_win position i corresponds to
        # w_head position (m-1-i), i.e. w position (m-1-i) as well (0-based).
        jac_row = np.zeros(n)
        for i in range(m):
            jac_row[m - 1 - i] = d_beta_d_s_index[i]
        batch_beta.append(b)
        batch_jac.append(jac_row)

        if len(batch_beta) == n_b:
            bb = np.array(batch_beta)               # (n_b,)
            jj = np.array(batch_jac)                 # (n_b, n)
            nb_eff = len(bb)
            # Eq. 13: piecewise-linear derivative of W1 objective wrt each
            # beta_{t+i}, using the EMPIRICAL rank pi-hat of the batch's own
            # beta values (sorted ascending), pushing each beta toward the
            # midpoint of its empirical-quantile bucket [ (k-1)/nb, k/nb ].
            order_b = np.argsort(bb, kind="mergesort")
            rank_b = np.empty(nb_eff, dtype=int)
            rank_b[order_b] = np.arange(nb_eff)       # 0-based rank
            pihat_1based = rank_b + 1                 # pi-hat(i), 1-based, for i = position in batch
            lower = (pihat_1based - 1) / nb_eff
            upper = pihat_1based / nb_eff
            dW1_dbeta = np.where(
                bb < lower, -1.0 / nb_eff,
                np.where(bb > upper, 1.0 / nb_eff,
                         2.0 * (bb - (pihat_1based - 1) / (2.0 * nb_eff)) / nb_eff)
            )
            grad_w = (dW1_dbeta[:, None] * jj).sum(axis=0)   # (n,)
            w = w - gamma * grad_w
            w = project(w)
            batch_beta = []
            batch_jac = []

    return beta, dict(n_c=n_c, n=n, n_b=n_b, gamma=gamma, alpha_c=alpha_c, w_final=w)


if __name__ == "__main__":
    # ---- sanity check 1: i.i.d. exchangeable normal scores -> beta ~ U[0,1],
    # detection rate at threshold alpha_c should be close to alpha_c. ----
    rng = np.random.default_rng(0)
    T = 3000
    alpha_c = 0.05
    scores_iid = rng.normal(size=T)
    beta, diag = w1_acas(scores_iid, alpha_c=alpha_c, n=300, n_b=20, gamma=0.05)
    valid = ~np.isnan(beta)
    flagged_rate = np.mean(beta[valid] < alpha_c)
    print(f"[sanity 1: i.i.d.] n_c={diag['n_c']}  valid={valid.sum()}/{T}  "
          f"flagged-rate={flagged_rate:.4f}  (target alpha_c={alpha_c})")

    # ---- sanity check 2: distribution shift (mean jumps up mid-sequence) ----
    scores_shift = np.concatenate([
        rng.normal(loc=0.0, size=T // 2),
        rng.normal(loc=2.0, size=T // 2),   # shifted upward -> more anomalies expected after t=T/2
    ])
    beta2, diag2 = w1_acas(scores_shift, alpha_c=alpha_c, n=300, n_b=20, gamma=0.05)
    valid2 = ~np.isnan(beta2)
    pre = beta2[:T // 2][valid2[:T // 2]]
    post = beta2[T // 2:][valid2[T // 2:]]
    print(f"[sanity 2: shift]  pre-shift flagged-rate={np.mean(pre < alpha_c):.4f}  "
          f"post-shift flagged-rate={np.mean(post < alpha_c):.4f}")
