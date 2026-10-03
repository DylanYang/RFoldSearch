# Ported from RFoldseek-paper-code src/search/evalue.py; only imports adjusted.
"""Cluster-maximum Gumbel E-values for RFoldseek token Smith-Waterman scores."""

from __future__ import annotations

import math
import warnings
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
    len_ratio=None,
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
    len_ratio=None,
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


def fit_query_parameter_model(lengths, entropies, mus, betas,
                              ridge: float = 1e-3) -> dict:
    """Fit ``[1, z(log length), z(token entropy)] -> (mu, log beta)``.

    The intercept is not penalised. Predictor bounds are stored so deployment
    can use the same boundary rule as calibration.
    """
    lengths = np.asarray(lengths, dtype=np.float64).reshape(-1)
    entropies = np.asarray(entropies, dtype=np.float64).reshape(-1)
    mus = np.asarray(mus, dtype=np.float64).reshape(-1)
    betas = np.asarray(betas, dtype=np.float64).reshape(-1)
    if not (len(lengths) == len(entropies) == len(mus) == len(betas)):
        raise ValueError("query-model inputs must have the same length")
    if len(lengths) < 3:
        raise ValueError("need at least three per-query Gumbel fits")
    if (np.any(~np.isfinite(lengths)) or np.any(lengths <= 0)
            or np.any(~np.isfinite(entropies))
            or np.any(~np.isfinite(mus))
            or np.any(~np.isfinite(betas)) or np.any(betas <= 0)):
        raise ValueError("query-model inputs must be finite and positive where required")
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and non-negative")

    log_length = np.log(lengths)
    length_scale = float(log_length.std(ddof=0))
    entropy_scale = float(entropies.std(ddof=0))
    if length_scale <= 0 or entropy_scale <= 0:
        raise ValueError("query-conditional predictors must vary in the training set")
    centre = {
        "log_length_mean": float(log_length.mean()),
        "log_length_scale": length_scale,
        "entropy_mean": float(entropies.mean()),
        "entropy_scale": entropy_scale,
    }
    design = np.column_stack([
        np.ones(len(lengths)),
        (log_length - centre["log_length_mean"]) / length_scale,
        (entropies - centre["entropy_mean"]) / entropy_scale,
    ])
    response = np.column_stack([mus, np.log(betas)])
    gram = design.T @ design
    gram[1:, 1:] += np.eye(2) * ridge
    coefficients = np.linalg.solve(gram, design.T @ response)
    return {
        "distribution": "Gumbel",
        "feature_order": ["intercept", "z_log_length", "z_token_entropy"],
        "centre": centre,
        "training_domain": {
            "log_length_min": float(log_length.min()),
            "log_length_max": float(log_length.max()),
            "entropy_min": float(entropies.min()),
            "entropy_max": float(entropies.max()),
        },
        "coefficients_mu_logbeta": coefficients.tolist(),
        "ridge": float(ridge),
    }


def calibrate_clustermax_query_conditional(
    sub_matrix: np.ndarray,
    token_lists,
    clusters,
    query_indices,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
    exclude=None,
    is_audited=None,
    audit_evalue_ceiling: float | None = None,
    ridge: float = 1e-3,
) -> dict:
    """Fit the deployment query model from full-database cluster maxima.

    ``query_indices`` selects a pre-defined calibration panel from the target
    database. For every query, all eligible target members are searched; the
    query's own cluster is omitted and exclusions are applied before taking a
    cluster maximum. Consequently, if a cluster's leading member is excluded,
    its best remaining member is used (cluster backfill).
    """
    sub = np.asarray(sub_matrix, dtype=np.float32)
    toks = [np.asarray(t, dtype=np.int64) for t in token_lists]
    clusters = np.asarray(clusters, dtype=object)
    if len(clusters) != len(toks):
        raise ValueError("clusters must give one label per chain")
    members: dict = {}
    for index, cluster in enumerate(clusters):
        members.setdefault(cluster, []).append(index)
    if len(members) < 2:
        raise ValueError("need at least two clusters to calibrate")
    query_indices = [int(index) for index in query_indices]
    if len(set(query_indices)) != len(query_indices):
        raise ValueError("query_indices must not contain duplicates")
    if any(index < 0 or index >= len(toks) for index in query_indices):
        raise ValueError("query index is outside the target database")

    rows = []
    for qi in query_indices:
        maxima = []
        for cluster, indices in members.items():
            if cluster == clusters[qi]:
                continue
            eligible = [index for index in indices
                        if exclude is None or not exclude(qi, index)]
            if not eligible:
                continue
            score = max(
                smith_waterman_numba(toks[qi], toks[index], sub,
                                     gap_open, gap_extend)
                for index in eligible
            )
            maxima.append(score)
        if len(maxima) < 20:
            raise ValueError(
                f"query index {qi} has too few eligible cluster maxima to fit")
        mu, beta = _gumbel_mle(maxima)
        counts = np.bincount(toks[qi])
        probabilities = counts[counts > 0] / len(toks[qi])
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        rows.append({
            "query_index": qi,
            "length": len(toks[qi]),
            "token_entropy": entropy,
            "mu": mu,
            "beta": beta,
            "n_maxima": len(maxima),
        })
    model = fit_query_parameter_model(
        [row["length"] for row in rows],
        [row["token_entropy"] for row in rows],
        [row["mu"] for row in rows],
        [row["beta"] for row in rows],
        ridge=ridge,
    )
    result = {
        "query_parameter_model": model,
        "query_fits": rows,
        "n_clusters": len(members),
        "n_chains": len(toks),
    }
    if is_audited is not None and audit_evalue_ceiling is not None:
        # Audit against the final deployed regression, not each query's
        # temporary MLE. Re-score instead of retaining every query--cluster
        # score in memory; calibration is offline and correctness matters more
        # than one additional scan.
        audit_params = {"query_parameter_model": model}
        missing_audit = []
        for qi in query_indices:
            mu, beta = clustermax_params_for_query(audit_params, toks[qi])
            for cluster, indices in members.items():
                if cluster == clusters[qi]:
                    continue
                eligible = [index for index in indices
                            if exclude is None or not exclude(qi, index)]
                if not eligible:
                    continue
                score, leader = max(
                    (smith_waterman_numba(toks[qi], toks[index], sub,
                                          gap_open, gap_extend), index)
                    for index in eligible
                )
                raw_evalue = evalue_clustermax(
                    score, mu, beta, len(members))
                if (raw_evalue <= audit_evalue_ceiling
                        and not is_audited(qi, leader)):
                    missing_audit.append({
                        "query_index": qi,
                        "target_index": int(leader),
                        "cluster": cluster,
                        "raw_score": float(score),
                        "raw_evalue": raw_evalue,
                    })
        result["missing_audit"] = missing_audit
    return result


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


def clustermax_params_for_query(params: dict, tokens):
    """Return Gumbel parameters for a query under the database's fitted model.

    Any database may store its own linear ridge model of per-query ``mu`` and
    ``log(beta)`` fitted from standardised log query length and the Shannon
    entropy (in bits) of its 3Dr-token composition. The historical length grid
    is used only when a database has no query-conditional model.
    """
    query = np.asarray(tokens, dtype=np.int64)
    if query.ndim != 1 or not len(query) or np.any(query < 0):
        raise ValueError("query tokens must be a nonempty 1D array of nonnegative IDs")
    model = params.get("query_parameter_model")
    if model is None:
        return clustermax_params_for_length(params, len(query))
    if model.get("distribution") != "Gumbel":
        raise ValueError("unsupported query-conditional E-value distribution")
    centre = model["centre"]
    domain = model.get("training_domain")
    if domain is None:
        raise ValueError("query-conditional E-value model needs a training_domain")
    log_min, log_max = (float(domain["log_length_min"]),
                        float(domain["log_length_max"]))
    entropy_min, entropy_max = (float(domain["entropy_min"]),
                                float(domain["entropy_max"]))
    if (not all(map(math.isfinite, (log_min, log_max, entropy_min, entropy_max)))
            or log_min > log_max or entropy_min > entropy_max):
        raise ValueError("invalid query-conditional E-value training_domain")
    counts = np.bincount(query)
    probabilities = counts[counts > 0] / len(query)
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    log_length = math.log(len(query))
    bounded_log_length = float(np.clip(log_length, log_min, log_max))
    bounded_entropy = float(np.clip(entropy, entropy_min, entropy_max))
    if bounded_log_length != log_length or bounded_entropy != entropy:
        warnings.warn(
            "query length or token entropy is outside the E-value model's "
            "training domain; boundary Gumbel parameters are used",
            RuntimeWarning, stacklevel=2)
    x = ((bounded_log_length - float(centre["log_length_mean"]))
         / float(centre["log_length_scale"]))
    h = ((bounded_entropy - float(centre["entropy_mean"]))
         / float(centre["entropy_scale"]))
    coefficients = np.asarray(model["coefficients_mu_logbeta"], dtype=np.float64)
    if coefficients.shape != (3, 2) or not np.all(np.isfinite(coefficients)):
        raise ValueError("query-conditional E-value coefficients must be finite 3x2")
    mu, log_beta = np.asarray([1.0, x, h]) @ coefficients
    beta = math.exp(float(log_beta))
    if not math.isfinite(mu) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("query-conditional Gumbel parameters must be finite and positive")
    return float(mu), beta


def evalue_clustermax(score, mu: float, beta: float, n_clusters: int):
    """Expected number of clusters reaching ``score`` by chance.

    The null was fitted to one maximum per cluster, so the search space is
    counted in clusters as well; a database that holds a family four hundred
    times still poses one question about it.
    """
    s = np.asarray(score, dtype=np.float64)
    # ``1 - exp(-x)`` loses all precision for very small ``x`` and used to
    # collapse the most significant hits to an exact zero.  ``-expm1(-x)`` is
    # algebraically identical but remains accurate throughout the tail.
    tail = -np.expm1(-np.exp(-(s - mu) / beta))
    return float(n_clusters) * tail


def fit_empirical_calibration(raw_evalues, weights=None, n_clusters=None) -> dict:
    """Fit a monotone empirical correction to background E-values.

    ``raw_evalues`` contains initial cluster-max E-values from known background
    (structurally unrelated) hits; the caller must first remove genuine
    structural homologues from the background -- same annotated family or clan,
    and cross-family candidates with a high symmetric US-align TM-score -- or
    the calibration is inflated by real matches and turns over-strict.
    ``weights`` gives each observation's contribution to the expected number of
    background clusters per query. Thus, at a threshold ``e``, the cumulative
    weight of observations with ``raw_evalue <= e`` is the observed
    background-cluster count to report as the calibrated E-value. Equal-query
    or equal-family weighting should be performed by the caller when creating
    these weights; equal-family weighting keeps large, redundant families from
    dominating the fit.

    Repeated knots are collapsed so tied raw E-values receive the same
    calibrated value. A half-observation finite-sample floor avoids unsupported
    zero E-values below the smallest observed background value. Values below
    the first knot are deliberately not extrapolated: the calibration data do
    not establish the shape of that unobserved tail.
    """
    values = np.asarray(raw_evalues, dtype=np.float64).reshape(-1)
    if weights is None:
        weights = np.ones(values.size, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("need at least one background E-value")
    if values.size != weights.size:
        raise ValueError("raw_evalues and weights must have the same size")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("raw_evalues must be finite and non-negative")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("weights must be finite and positive")
    if n_clusters is not None and int(n_clusters) < 1:
        raise ValueError("n_clusters must be positive")

    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    knots, first = np.unique(values, return_index=True)
    knot_weights = np.add.reduceat(weights, first)
    cumulative = np.cumsum(knot_weights)
    result = {
        "scheme": "empirical-monotone",
        "raw_evalue_knots": knots.tolist(),
        "calibrated_evalues": cumulative.tolist(),
        "finite_sample_floor": float(0.5 * weights.min()),
        "n_observations": int(values.size),
        "total_weight": float(cumulative[-1]),
    }
    if n_clusters is not None:
        result["n_clusters"] = int(n_clusters)
    return result


def thin_empirical_calibration(calibration: dict, max_relative_step: float = 0.01) -> dict:
    """Merge adjacent knots of an empirical map into conservative segments.

    Each kept knot starts a segment of the original step function and takes the
    largest calibrated value inside it, so the thinned map never reports a
    smaller E-value than the full one, and a segment ends before the full map
    rises more than ``max_relative_step`` above its first value. The first knot,
    the finite-sample floor, the final value and any ceiling are unchanged.
    """
    if not math.isfinite(max_relative_step) or max_relative_step < 0:
        raise ValueError("max_relative_step must be finite and non-negative")
    knots = np.asarray(calibration["raw_evalue_knots"], dtype=np.float64)
    mapped = np.asarray(calibration["calibrated_evalues"], dtype=np.float64)
    keep_knots, keep_values = [], []
    start = 0
    while start < len(knots):
        end = int(np.searchsorted(mapped, mapped[start] * (1.0 + max_relative_step),
                                  side="right"))
        end = max(end, start + 1)
        keep_knots.append(float(knots[start]))
        keep_values.append(float(mapped[end - 1]))
        start = end
    thinned = dict(calibration)
    thinned["raw_evalue_knots"] = keep_knots
    thinned["calibrated_evalues"] = keep_values
    thinned["thinning"] = {"max_relative_step": float(max_relative_step),
                           "n_full_knots": int(len(knots)),
                           "rule": "segment maximum (conservative)"}
    return thinned


def apply_empirical_calibration(raw_evalue, calibration: dict):
    """Map initial E-values to observed background clusters per query.

    The empirical CDF is a right-continuous step function: a raw E-value equal
    to a stored knot includes that knot. Values below the first knot receive
    the finite-sample floor rather than an unvalidated tail extrapolation.
    A calibration fitted only up to an explicit raw
    E-value ceiling reports the uncalibrated (conservative) raw E-value beyond
    that ceiling instead of saturating at the last empirical count.
    """
    values = np.asarray(raw_evalue, dtype=np.float64)
    knots = np.asarray(calibration["raw_evalue_knots"], dtype=np.float64)
    mapped = np.asarray(calibration["calibrated_evalues"], dtype=np.float64)
    if knots.ndim != 1 or mapped.ndim != 1 or len(knots) != len(mapped) or not len(knots):
        raise ValueError("invalid empirical E-value calibration table")
    if np.any(np.diff(knots) <= 0) or np.any(np.diff(mapped) < 0):
        raise ValueError("empirical E-value calibration must be monotone")
    positions = np.searchsorted(knots, values, side="right")
    floor = float(calibration.get("finite_sample_floor", 0.0))
    output = np.full(values.shape, floor, dtype=np.float64)
    present = positions > 0
    output[present] = mapped[np.minimum(positions[present] - 1, len(mapped) - 1)]
    ceiling = calibration.get("candidate_raw_evalue_ceiling")
    if ceiling is not None:
        if float(ceiling) < float(knots[-1]):
            raise ValueError("empirical E-value ceiling is below the final knot")
        output = np.where(values > float(ceiling),
                          np.maximum(values, float(mapped[-1])), output)
    if output.ndim == 0:
        return float(output)
    return output


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
