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
``lambda``/``K``; we calibrate them once by scoring pairs of real database
chains, with the *same* matrix and gap penalties as the search, censoring the
highest-scoring fraction (which is where genuinely related chains land) and
fitting a Gumbel distribution to the remainder. Randomised strings must not be
used for this: shuffling or resampling tokens destroys the generic helical
grammar shared by all RNA structures, putting the null far below real unrelated
chains and inflating significance by orders of magnitude. Calibration is done by build_evalue_params.py and cached
to files/models/evalue_params.json; the RFoldseek CLI only reads that file.
"""

from __future__ import annotations

import math
import numpy as np

from .smith_waterman import smith_waterman_numba

EULER_GAMMA = 0.5772156649015329


def _gumbel_mle(scores):
    """Maximum-likelihood Gumbel fit; returns (mu, beta)."""
    x = np.asarray(scores, dtype=np.float64)
    beta = max(np.std(x, ddof=1) * math.sqrt(6.0) / math.pi, 1e-6)
    for _ in range(200):                      # Newton on the profile likelihood
        z = np.exp(-x / beta)
        sz, sxz = z.sum(), (x * z).sum()
        f = x.mean() - sxz / sz - beta
        if abs(f) < 1e-10:
            break
        sxxz = (x * x * z).sum()
        df = (sxz * sxz) / (sz * sz * beta * beta) - sxxz / (sz * beta * beta) - 1.0
        step = f / df if abs(df) > 1e-12 else 0.0
        beta = max(beta - step, 1e-6)
    mu = -beta * math.log(np.exp(-x / beta).mean())
    return float(mu), float(beta)


def calibrate(
    sub_matrix: np.ndarray,
    token_lists,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
    length: int = 300,
    n_pairs: int = 4000,
    seed: int = 0,
    censor: float = 0.10,
) -> dict:
    """Fit Karlin-Altschul lambda/K from REAL database chain pairs.

    The null must describe what an *unrelated real RNA* scores, not what a
    randomised string scores. Sampling tokens independently from the database
    background composition -- as this function previously did -- destroys the
    generic helical grammar every RNA structure shares, so the null sits far
    below real unrelated chains and the resulting E-values overstate
    significance by orders of magnitude.

    Query chains of approximately ``length`` tokens are therefore aligned
    against uniformly drawn database chains. A redundant database contributes
    some genuinely related chains to such a sample, so the highest-scoring
    ``censor`` fraction is dropped before fitting (Pearson, 1998). The
    remaining scores are fitted by maximum likelihood and converted to the
    Karlin-Altschul parameterisation via beta = 1/lambda and
    mu = ln(K m n) / lambda.
    """
    rng = np.random.default_rng(seed)
    sub = np.asarray(sub_matrix, dtype=np.float32)
    toks = [np.asarray(t, dtype=np.int64) for t in token_lists if len(t) >= 4]
    if not toks:
        raise ValueError("no usable chains for calibration")
    lens = np.array([len(t) for t in toks])

    # queries: the chains closest in length to the requested calibration point
    order = np.argsort(np.abs(lens - float(length)))
    pool = order[: max(8, min(len(order), 64))]

    scores = np.empty(n_pairs, dtype=np.float64)
    q_len = np.empty(n_pairs)
    t_len = np.empty(n_pairs)
    for i in range(n_pairs):
        qi = int(pool[rng.integers(len(pool))])
        ti = int(rng.integers(len(toks)))
        while ti == qi:
            ti = int(rng.integers(len(toks)))
        scores[i] = smith_waterman_numba(toks[qi], toks[ti], sub, gap_open, gap_extend)
        q_len[i], t_len[i] = len(toks[qi]), len(toks[ti])

    keep = np.argsort(scores)[: max(10, int(round(n_pairs * (1.0 - censor))))]
    kept = scores[keep]
    mu, beta = _gumbel_mle(kept)
    lam = 1.0 / beta
    m_eff, n_eff = float(q_len[keep].mean()), float(t_len[keep].mean())
    k_param = float(math.exp(lam * mu) / (m_eff * n_eff))
    return {
        "lambda": float(lam),
        "K": k_param,
        "length": int(length),
        "n_pairs": int(n_pairs),
        "censor": float(censor),
        "m_eff": m_eff,
        "n_eff": n_eff,
        "null_score_mean": float(kept.mean()),
        "null_score_std": float(kept.std(ddof=1)),
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
    token_lists,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
    lengths=(20, 30, 40, 60, 80, 120, 160, 220, 300, 500, 1000, 2000),
    n_pairs: int = 4000,
    long_n_pairs: int = 4000,
    long_thresh: int = 300,
    seed: int = 0,
    monotone: bool = False,
) -> list:
    """Calibrate lambda/K on a grid of sequence lengths.

    Asymptotic Karlin-Altschul theory is length-independent, but at finite
    length the random local-alignment score distribution has an edge effect,
    so lambda/K are calibrated at several lengths and the caller interpolates
    to the actual query length. Since ~two thirds of the database chains are
    < 100 nt, the grid is dense at the short end.

    Measured on the 885-chain benchmark, lambda falls with length only over
    the short arm, reaching about 0.38 near L=120, and then *rises* again to
    about 0.51 by L=1000, where it flattens. Repeating each point over three
    seeds puts the seed-to-seed spread at +/-0.005, fifteen times smaller than
    that rise, so the shape is real and not sampling noise. ``monotone`` is
    therefore off by default: forcing the grid to be non-increasing collapses
    every point above L=80 onto one value and discards a genuine trend. With
    lambda pinned 15% low at L=1000, and E proportional to exp(-lambda S),
    a score of 60 comes out roughly two orders of magnitude too conservative
    -- and long chains are the ribosomal targets a search hits most often.
    Set ``monotone=True`` to restore the old smoothing; raw values are kept as
    ``lambda_raw``/``K_raw`` whenever it is on.

    ``long_n_pairs`` applies above ``long_thresh``, where the O(L^2) alignment
    is expensive. It matches ``n_pairs`` by default: at 800 pairs the L=500
    estimate was off by 6% against the 4000-pair value, and one length point
    costs about 30 seconds.
    """
    table = []
    for L in lengths:
        npair = n_pairs if L <= long_thresh else long_n_pairs
        par = calibrate(sub_matrix, token_lists, gap_open, gap_extend, int(L), npair, seed)
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
    # Sanity on synthetic "chains" with local structure: unrelated pairs should
    # be unremarkable, a self-alignment overwhelming.
    rng = np.random.default_rng(0)
    n_codes = 80
    sub = rng.standard_normal((n_codes, n_codes)).astype(np.float32) * 0.4
    sub = (sub + sub.T) / 2
    np.fill_diagonal(sub, 1.0)
    chains = []
    for _ in range(60):                     # runs of repeated tokens, as real
        out = []                            # token strings are autocorrelated
        while len(out) < 200:
            out += [int(rng.integers(n_codes))] * int(rng.integers(1, 5))
        chains.append(np.asarray(out[:200], dtype=np.int64))
    par = calibrate(sub, chains, length=200, n_pairs=400, seed=1)
    lam, kp = par["lambda"], par["K"]
    print("lambda=%.4f K=%.4g null mean=%.2f std=%.2f" %
          (lam, kp, par["null_score_mean"], par["null_score_std"]))
    N = 200 * 500
    for name, (a, b) in {"unrelated": (chains[0], chains[1]),
                         "identical": (chains[0], chains[0])}.items():
        sc = smith_waterman_numba(a, b, sub, -2, -0.5)
        print("%-10s S=%.1f  E=%.3g" % (name, sc, evalue(sc, 200, N, lam, kp)))
