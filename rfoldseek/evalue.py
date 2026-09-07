# Ported from RFoldseek-paper-code src/search/evalue.py; only imports adjusted.
"""Karlin-Altschul E-values for RFoldseek token Smith-Waterman scores.

RFoldseek ranks candidates by a raw local Smith-Waterman score under the
Structure-BLOSUM matrix. A raw score is not on its own interpretable: the user
cannot tell whether SW=30 is significant. As in BLAST/Foldseek it is converted
to an E-value, the expected number of chance hits at least this good:

    E = N_clusters * (1 - exp(-exp(-(S - mu) / beta)))

The null is fitted to *cluster maxima*. A Gumbel is the limit law of a
maximum, so the sample it is fitted to should be maxima; taking the best score
a query reaches against each sequence cluster makes that true. It also makes
redundancy harmless, which is the property a structure database needs: a
family present four hundred times contributes one draw to the null and counts
as one question of the database, not four hundred. The alternative -- fitting
individual chain pairs and censoring the top fraction where near-duplicates
land -- needs a censoring fraction retuned per database, and on a redundant
database is anti-conservative by two orders of magnitude at small E.

Clusters come from ``cd-hit-est -c 0.8 -n 5 -aS 0.8 -aL 0.8 -g 1`` upstream;
nothing here clusters anything itself. Randomised strings must not be used for
the null: shuffling or resampling tokens destroys the generic helical grammar
shared by all RNA structures, putting the null far below real unrelated chains
and inflating significance by orders of magnitude.

Calibration runs at build time (scripts/rebuild_evalue_params.py) and is
cached to assets/evalue_params.json; the search path only reads that file, so
E-values add nothing to query time.
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


def calibrate_clustermax(
    sub_matrix: np.ndarray,
    token_lists,
    clusters,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
    length: int = 300,
    n_queries: int = 64,
    seed: int = 0,
    exclude=None,
    len_ratio: float = 3.0,
) -> dict:
    """Fit the null to cluster maxima rather than to individual pairs.

    A Gumbel is the limit law of a maximum, so the sample it is fitted to
    should be maxima. Taking the best score a query reaches against each
    sequence cluster makes that true, and at the same time makes redundancy
    harmless: a family present four hundred times contributes one draw, not
    four hundred. Nothing then has to be censored to keep near-duplicates from
    dominating the tail, which removes the one parameter that had to be
    retuned for every database.

    ``clusters`` gives a cluster label per chain, as produced upstream by
    ``cd-hit-est -c 0.8 -n 5 -aS 0.8 -aL 0.8 -g 1``; this function does not
    cluster anything itself. ``exclude(i, j)`` may reject a query/target pair
    on labels the caller holds, such as the same Rfam family or clan.

    Returns ``mu``/``beta`` of the fitted Gumbel together with the cluster and
    chain counts; feed ``mu``, ``beta`` and the cluster count to
    ``evalue_clustermax``.
    """
    rng = np.random.default_rng(seed)
    sub = np.asarray(sub_matrix, dtype=np.float32)
    toks = [np.asarray(t, dtype=np.int64) for t in token_lists]
    clusters = np.asarray(clusters, dtype=object)
    if len(clusters) != len(toks):
        raise ValueError("clusters must give one label per chain")
    members: dict = {}
    for i, c in enumerate(clusters):
        members.setdefault(c, []).append(i)
    if len(members) < 2:
        raise ValueError("need at least two clusters to calibrate")

    lens = np.array([len(t) for t in toks])
    order = np.argsort(np.abs(lens - float(length)))
    pool = [int(i) for i in order[: max(2, min(len(order), n_queries))]]

    maxima = []
    for qi in pool:
        for c, mem in members.items():
            if c == clusters[qi]:
                continue
            mem = [j for j in mem if j != qi and (exclude is None or not exclude(qi, j))]
            if len_ratio is not None:
                m_q = len(toks[qi])
                mem = [j for j in mem
                       if len(toks[j]) <= len_ratio * m_q and m_q <= len_ratio * len(toks[j])]
            if not mem:
                continue
            maxima.append(max(smith_waterman_numba(toks[qi], toks[j], sub,
                                                   gap_open, gap_extend) for j in mem))
    x = np.asarray(maxima, dtype=np.float64)
    if x.size < 20:
        raise ValueError("too few cluster maxima to fit; widen the query pool")
    mu, beta = _gumbel_mle(x)
    return {
        "mu": float(mu),
        "beta": float(beta),
        "lambda": float(1.0 / beta),
        "n_clusters": len(members),
        "n_chains": len(toks),
        "n_maxima": int(x.size),
        "length": int(length),
        "null_score_mean": float(x.mean()),
        "null_score_std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
    }


def calibrate_clustermax_length_dependent(
    sub_matrix: np.ndarray,
    token_lists,
    clusters,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
    lengths=(20, 30, 40, 60, 80, 120, 160, 220, 300, 500, 1000, 2000),
    n_queries: int = 64,
    seed: int = 0,
    exclude=None,
    len_ratio: float = 3.0,
    max_length_gap: float = 2.0,
) -> dict:
    """Cluster-max calibration on a grid of query lengths.

    Grid points the database cannot supply are skipped rather than filled from
    whatever chain happens to be nearest: a point is used only when the median
    query in its pool is within ``max_length_gap`` of it. Filling instead
    produces rows that repeat a neighbouring fit and read, in the saved
    parameters, as though they had been calibrated.

    Returns the table alongside the cluster count and the skipped lengths.
    """
    toks = [np.asarray(t, dtype=np.int64) for t in token_lists]
    lens = np.array([len(t) for t in toks])
    table, skipped = [], []
    n_clusters = len(set(np.asarray(clusters, dtype=object).tolist()))
    for L in lengths:
        order = np.argsort(np.abs(lens - float(L)))
        pool = lens[order[: max(2, min(len(order), n_queries))]]
        med = float(np.median(pool))
        if med > max_length_gap * L or L > max_length_gap * med:
            skipped.append(int(L))
            continue
        par = calibrate_clustermax(sub_matrix, toks, clusters, gap_open, gap_extend,
                                   int(L), n_queries, seed, exclude, len_ratio)
        table.append({"length": int(L), "mu": par["mu"], "beta": par["beta"],
                      "lambda": par["lambda"], "n_maxima": par["n_maxima"]})
    if not table:
        raise ValueError("no grid length is represented in this database")
    return {"length_table": table, "n_clusters": n_clusters,
            "n_chains": len(toks), "skipped_lengths": skipped}


def clustermax_params_for_length(params: dict, m: float):
    """Interpolate ``(mu, beta)`` for a query of length ``m``.

    Both are interpolated linearly and clamped to the ends of the grid.
    """
    tbl = params["length_table"]
    Ls = np.array([r["length"] for r in tbl], dtype=np.float64)
    mus = np.array([r["mu"] for r in tbl], dtype=np.float64)
    betas = np.array([r["beta"] for r in tbl], dtype=np.float64)
    mm = float(np.clip(m, Ls[0], Ls[-1]))
    return float(np.interp(mm, Ls, mus)), float(np.interp(mm, Ls, betas))


def evalue_clustermax(score, mu: float, beta: float, n_clusters: int):
    """Expected number of clusters reaching ``score`` by chance.

    The null was fitted to one maximum per cluster, so the search space is
    counted in clusters as well; a database that holds a family four hundred
    times still poses one question about it.
    """
    s = np.asarray(score, dtype=np.float64)
    return float(n_clusters) * (1.0 - np.exp(-np.exp(-(s - mu) / beta)))


def bit_score_clustermax(score, mu: float, beta: float):
    """Structural bit score S' = (S - mu) / (beta * ln 2).

    Normalises the raw token Smith-Waterman score to bits, so it no longer
    carries the arbitrary scale of the Structure-BLOSUM matrix and is
    comparable across queries. It is the cluster-max counterpart of a
    BLAST/Foldseek bit score and satisfies E = N_clusters * 2**(-S') in the
    tail, where the Gumbel survival function is exp(-(S - mu) / beta).
    """
    s = np.asarray(score, dtype=np.float64)
    return (s - mu) / (beta * np.log(2.0))


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
    clusters = [i // 3 for i in range(len(chains))]   # three near-duplicates each
    par = calibrate_clustermax(sub, chains, clusters, length=200,
                               n_queries=20, seed=1)
    print("mu=%.2f beta=%.2f  clusters=%d chains=%d" %
          (par["mu"], par["beta"], par["n_clusters"], par["n_chains"]))
    for name, (a, b) in {"unrelated": (chains[0], chains[5]),
                         "identical": (chains[0], chains[0])}.items():
        sc = smith_waterman_numba(a, b, sub, -2, -0.5)
        print("%-10s S=%.1f  bits=%.1f  E=%.3g"
              % (name, sc, bit_score_clustermax(sc, par["mu"], par["beta"]),
                 evalue_clustermax(sc, par["mu"], par["beta"], par["n_clusters"])))
