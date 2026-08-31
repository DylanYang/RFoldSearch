# Ported from RFoldseek-paper-code src/search/evalue.py; only imports adjusted.
"""Karlin-Altschul E-values for RFoldseek token Smith-Waterman scores.

RFoldseek reranks candidates by a raw local Smith-Waterman score under the
Structure-BLOSUM matrix. A raw score is not, on its own, interpretable: the user
cannot tell whether SW=30 is significant. As in BLAST/Foldseek we convert it to
an E-value, the expected number of database hits scoring at least this high by
chance:

    E = K * m * N * exp(-lambda * S)

where ``m`` is the query length, ``N`` is the total number of tokens in the
target database, and ``lambda``/``K`` are the Gumbel parameters of the null
score distribution. Gapped local alignment has no closed form for
``lambda``/``K``; we calibrate them once by scoring random token-sequence pairs
drawn from the database's background token frequencies, with the *same* matrix
and gap penalties as the search, and fitting a Gumbel distribution to the
optimal local scores. Calibration is done by build_evalue_params.py and cached
to files/models/evalue_params.json; the RFoldseek CLI only reads that file.
"""

from __future__ import annotations

import numpy as np

from .smith_waterman import smith_waterman_numba

EULER_GAMMA = 0.5772156649015329


def background_frequencies(token_lists, n_codes: int) -> np.ndarray:
    """Empirical token frequencies over the whole database (Laplace-smoothed)."""
    counts = np.ones(n_codes, dtype=np.float64)  # +1 smoothing avoids p == 0
    for t in token_lists:
        if len(t):
            counts += np.bincount(np.asarray(t, dtype=np.int64), minlength=n_codes)[:n_codes]
    return counts / counts.sum()


def _fit_gumbel_mle(scores, max_iter: int = 200, tol: float = 1e-10):
    """Maximum-likelihood Gumbel ``(lambda, mu)`` for local-alignment scores.

    The optimal local score of random sequence pairs is asymptotically Gumbel;
    MLE uses the whole shape of the score distribution (not just its variance),
    so it is statistically more efficient and less sensitive to the upper tail
    than the method-of-moments estimator. Profiling out the location ``mu``
    leaves a one-dimensional score equation for ``lambda``,

        1/lambda = mean(s) - <s>_w,   <s>_w = sum(s e^{-lambda s}) / sum(e^{-lambda s}),

    whose right-hand side is monotone, so a sign-changing bracket around the
    method-of-moments lambda is bisected to convergence. The location then
    follows in closed form from d(logL)/d(mu) = 0,

        mu = s_min + (1/lambda) * ln( n / sum_i e^{-lambda (s_i - s_min)} ).

    Weights are computed relative to ``s_min`` so ``exp`` never overflows.
    Returns ``(lambda, mu)``; the Karlin-Altschul K is ``exp(lambda*mu)/(m n)``.
    """
    s = np.asarray(scores, dtype=np.float64)
    n = s.size
    smean = float(s.mean())
    smin = float(s.min())
    std = float(s.std(ddof=1))
    if std <= 0:
        raise ValueError("degenerate null score distribution (std=0); check matrix/gaps")

    def g(lam):
        w = np.exp(-lam * (s - smin))          # in (0, 1], = 1 at s_min
        return 1.0 / lam - smean + float((s * w).sum() / w.sum())

    lam0 = float(np.pi / (std * np.sqrt(6.0)))  # method-of-moments initialiser
    lo, hi = lam0 * 0.2, lam0 * 5.0
    glo, ghi = g(lo), g(hi)
    it = 0
    while glo * ghi > 0.0 and it < 80:          # widen until the root is bracketed
        lo *= 0.5
        hi *= 1.5
        glo, ghi = g(lo), g(hi)
        it += 1
    if glo * ghi > 0.0:
        return lam0, smin + (1.0 / lam0) * float(
            np.log(n / np.exp(-lam0 * (s - smin)).sum()))
    lam = lam0
    for _ in range(max_iter):                   # bisection (g is monotone in lam)
        lam = 0.5 * (lo + hi)
        gm = g(lam)
        if abs(gm) < tol or (hi - lo) < tol * lam:
            break
        if glo * gm <= 0.0:
            hi, ghi = lam, gm
        else:
            lo, glo = lam, gm
    w = np.exp(-lam * (s - smin))
    mu = smin + (1.0 / lam) * float(np.log(n / w.sum()))
    return float(lam), float(mu)


def calibrate(
    sub_matrix: np.ndarray,
    background: np.ndarray,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
    length: int = 300,
    n_pairs: int = 4000,
    seed: int = 0,
    method: str = "mle",
) -> dict:
    """Fit Karlin-Altschul lambda/K from random token-pair SW scores.

    Random sequences of ``length`` tokens are drawn i.i.d. from ``background``
    and aligned with the deployed matrix / gap penalties. The optimal local
    score of independent sequences is asymptotically Gumbel-distributed:

        E[S]  = (ln(K m n) + gamma) / lambda,
        Var[S] = pi^2 / (6 lambda^2).

    Method-of-moments estimator (Altschul & Gish, 1996):
        lambda = pi / (std(S) * sqrt(6)),
        K      = exp(lambda * mean(S) - gamma) / (m * n).
    """
    n_codes = sub_matrix.shape[0]
    rng = np.random.default_rng(seed)
    sub = np.asarray(sub_matrix, dtype=np.float32)
    p = np.asarray(background, dtype=np.float64)
    p = p / p.sum()

    scores = np.empty(n_pairs, dtype=np.float64)
    for i in range(n_pairs):
        a = rng.choice(n_codes, size=length, p=p).astype(np.int64)
        b = rng.choice(n_codes, size=length, p=p).astype(np.int64)
        scores[i] = smith_waterman_numba(a, b, sub, gap_open, gap_extend)

    mean = float(scores.mean())
    std = float(scores.std(ddof=1))
    if std <= 0:
        raise ValueError("degenerate null score distribution (std=0); check matrix/gaps")
    mn = float(length) * float(length)
    if method == "mle":
        # Maximum-likelihood Gumbel fit (uses the whole score distribution).
        lam, mu = _fit_gumbel_mle(scores)
        k_param = float(np.exp(lam * mu) / mn)
    elif method == "mom":
        # Method-of-moments (Altschul & Gish, 1996): variance -> lambda, mean -> K.
        lam = float(np.pi / (std * np.sqrt(6.0)))
        k_param = float(np.exp(lam * mean - EULER_GAMMA) / mn)
    else:
        raise ValueError(f"unknown method {method!r}; use 'mle' or 'mom'")
    return {
        "lambda": lam,
        "K": k_param,
        "method": method,
        "gap_open": float(gap_open),
        "gap_extend": float(gap_extend),
        "calib_length": int(length),
        "n_pairs": int(n_pairs),
        "seed": int(seed),
        "null_score_mean": mean,
        "null_score_std": std,
        "null_score_min": float(scores.min()),
        "note": (
            "Karlin-Altschul Gumbel parameters (method-of-moments) for the token "
            "Smith-Waterman score under Structure-BLOSUM. Database E-value for a "
            "hit scoring S is E = K * m * N * exp(-lambda * S), with m the query "
            "length and N the total token count of the target database."
        ),
    }


def _isotonic_nonincreasing(y) -> np.ndarray:
    """Least-squares non-increasing fit (pool-adjacent-violators).

    lambda and K are both physically non-increasing in sequence length; PAVA
    removes the finite-sample wobble of the method-of-moments estimates (e.g. a
    single length whose random draw happens to score high) while staying as
    close as possible to the raw values.
    """
    y = np.asarray(y, dtype=np.float64)
    vals, wts, cnts = [], [], []
    for v in y:
        w, c = 1.0, 1
        while vals and vals[-1] < v:  # violates non-increasing -> pool
            pv, pw, pc = vals.pop(), wts.pop(), cnts.pop()
            v = (v * w + pv * pw) / (w + pw)
            w += pw
            c += pc
        vals.append(v); wts.append(w); cnts.append(c)
    out = []
    for v, c in zip(vals, cnts):
        out.extend([v] * c)
    return np.asarray(out, dtype=np.float64)


def calibrate_length_dependent(
    sub_matrix: np.ndarray,
    background: np.ndarray,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
    lengths=(20, 30, 40, 60, 80, 120, 160, 220, 300, 500, 1000, 2000),
    n_pairs: int = 4000,
    long_n_pairs: int = 800,
    long_thresh: int = 300,
    seed: int = 0,
    monotone: bool = True,
    method: str = "mle",
) -> list:
    """Calibrate lambda/K on a grid of sequence lengths.

    Asymptotic Karlin-Altschul theory is length-independent, but at finite
    length the random local-alignment score distribution has an edge effect:
    short sequences give a *larger* lambda/K and long sequences a smaller one,
    so a single fixed calibration over-estimates significance for long chains
    (rRNA) and under-estimates it for short ones. Since ~two thirds of the
    database chains are < 100 nt, we calibrate lambda/K at several lengths and
    let the caller interpolate to the actual query length. Long lengths
    (> ``long_thresh``) use fewer random pairs: lambda/K are essentially
    converged there and the O(L^2) alignment is expensive. With
    ``monotone=True`` the lambda and log-K grids are smoothed to be
    non-increasing (both are physically monotone in length), removing
    method-of-moments wobble; raw values are retained as ``lambda_raw``/``K_raw``.
    """
    table = []
    for L in lengths:
        npair = n_pairs if L <= long_thresh else long_n_pairs
        par = calibrate(sub_matrix, background, gap_open, gap_extend, int(L), npair, seed, method)
        table.append({
            "length": int(L),
            "lambda": par["lambda"],
            "K": par["K"],
            "n_pairs": npair,
            "null_score_mean": par["null_score_mean"],
            "null_score_std": par["null_score_std"],
        })
    if monotone and len(table) > 1:
        lam_s = _isotonic_nonincreasing([r["lambda"] for r in table])
        k_s = np.exp(_isotonic_nonincreasing(np.log([r["K"] for r in table])))
        for r, lam, kk in zip(table, lam_s, k_s):
            r["lambda_raw"], r["K_raw"] = r["lambda"], r["K"]
            r["lambda"], r["K"] = float(lam), float(kk)
    return table


def params_for_length(params: dict, m: float):
    """Interpolate (lambda, K) for a query of length ``m`` from a length table.

    Falls back to the scalar ``lambda``/``K`` when the params dict has no
    ``length_table`` (backward compatible with the fixed-length calibration).
    K is interpolated in log space (it spans orders of magnitude); both are
    clamped to the calibration-grid endpoints.
    """
    tbl = params.get("length_table")
    if not tbl:
        return float(params["lambda"]), float(params["K"])
    Ls = np.array([r["length"] for r in tbl], dtype=np.float64)
    lams = np.array([r["lambda"] for r in tbl], dtype=np.float64)
    logK = np.log(np.array([r["K"] for r in tbl], dtype=np.float64))
    mm = float(np.clip(m, Ls[0], Ls[-1]))
    lam = float(np.interp(mm, Ls, lams))
    kk = float(np.exp(np.interp(mm, Ls, logK)))
    return lam, kk


def evalue(score, m, n, lam: float, k_param: float):
    """Karlin-Altschul E-value E = K*m*n*exp(-lambda*S).

    For a database search, pass ``n`` = total tokens in the target database to
    obtain the database E-value (expected number of chance hits this good).
    ``score`` may be a scalar or an array.
    """
    s = np.asarray(score, dtype=np.float64)
    return k_param * float(m) * float(n) * np.exp(-lam * s)


def bit_score(score, m_unused, lam: float, k_param: float):
    """Structural bit score S' = (lambda*S - ln K) / ln 2.

    Normalises the raw token Smith-Waterman score to bits, independent of the
    Structure-BLOSUM matrix scale, so it is comparable across queries (the
    structural analogue of a BLAST/Foldseek bit score). Related to the E-value
    by E = m * N * 2**(-S'). ``m_unused`` is accepted for a uniform call
    signature with ``evalue`` but does not enter the bit score itself.
    """
    s = np.asarray(score, dtype=np.float64)
    return (lam * s - np.log(k_param)) / np.log(2.0)


def neg_log10_evalue(score, m, n, lam: float, k_param: float):
    """-log10(E); larger = more significant. Numerically safe for tiny E."""
    s = np.asarray(score, dtype=np.float64)
    return (lam * s) / np.log(10.0) - np.log10(max(k_param * float(m) * float(n), 1e-300))


if __name__ == "__main__":
    # Sanity: random pairs -> E ~ O(m*n) (not significant); identical -> E << 1.
    rng = np.random.default_rng(0)
    n_codes = 80
    sub = rng.standard_normal((n_codes, n_codes)).astype(np.float32) * 0.4
    sub = (sub + sub.T) / 2
    np.fill_diagonal(sub, 1.0)
    bg = np.ones(n_codes) / n_codes
    par = calibrate(sub, bg, length=200, n_pairs=1500, seed=1)
    lam, kp = par["lambda"], par["K"]
    print("lambda=%.4f K=%.4g null mean=%.2f std=%.2f" %
          (lam, kp, par["null_score_mean"], par["null_score_std"]))
    a = rng.choice(n_codes, size=200, p=bg).astype(np.int64)
    b = rng.choice(n_codes, size=200, p=bg).astype(np.int64)
    N = 200 * 500  # pretend a 500-chain database of length-200 chains
    print("random  : S=%.1f  E=%.3g" % (smith_waterman_numba(a, b, sub, -2, -0.5),
          evalue(smith_waterman_numba(a, b, sub, -2, -0.5), 200, N, lam, kp)))
    print("identical: S=%.1f  E=%.3g" % (smith_waterman_numba(a, a, sub, -2, -0.5),
          evalue(smith_waterman_numba(a, a, sub, -2, -0.5), 200, N, lam, kp)))
