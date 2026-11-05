"""Shared target-aware cluster E-value model and empirical calibration."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


SCHEME = "target-aware-raw-sw-empirical-v1"
FEATURES = ("log_query_length", "log_cluster_median_length",
            "log_length_difference", "token_entropy")


def token_entropy(tokens) -> float:
    values = np.asarray(tokens, dtype=np.int64)
    if values.ndim != 1 or not len(values) or np.any(values < 0):
        raise ValueError("query tokens must be a nonempty 1D array")
    counts = np.bincount(values)
    p = counts[counts > 0] / len(values)
    # Match the natural-log token entropy used in the frozen LOFO comparison.
    return float(-(p * np.log(p)).sum())


def feature_rows(query_lengths, entropies, cluster_lengths):
    qlog = np.log(np.asarray(query_lengths, dtype=float))[:, None]
    clog = np.log(np.asarray(cluster_lengths, dtype=float))[None, :]
    return np.stack(np.broadcast_arrays(qlog, clog, np.abs(qlog - clog),
                                        np.asarray(entropies, dtype=float)[:, None]),
                    axis=-1)


def _family_weights(families):
    _, inverse, sizes = np.unique(families, return_inverse=True,
                                  return_counts=True)
    return 1.0 / (len(sizes) * sizes[inverse])


def _weighted_quantile(values, weights, probability):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    place = min(int(np.searchsorted(cumulative, probability * cumulative[-1])),
                len(order) - 1)
    return float(values[order[place]])


def fit_gumbel(scores, leaders, query_lengths, entropies, cluster_lengths,
               families, *, censored=True, query_weights=None, compress=False):
    """Fit the validated five-feature Gumbel likelihood to clean cluster maxima."""
    scores = np.asarray(scores, dtype=float)
    leaders = np.asarray(leaders, dtype=int)
    query_lengths = np.asarray(query_lengths, dtype=float)
    cluster_lengths = np.asarray(cluster_lengths, dtype=float)
    families = np.asarray(families)
    if (scores.ndim != 2 or leaders.shape[0] != scores.shape[0]
            or leaders.shape[1] != len(cluster_lengths)
            or len(query_lengths) != scores.shape[0]
            or len(entropies) != scores.shape[0]
            or len(families) != scores.shape[0]):
        raise ValueError("score, leader and query metadata dimensions differ")
    rows, cols = np.where(leaders >= 0)
    if len(rows) < 100:
        raise ValueError("too few eligible cluster maxima for Gumbel fitting")
    targets = leaders[rows, cols]
    if np.any(targets >= scores.shape[1]):
        raise ValueError("cluster leader index is outside the score matrix")
    y = scores[rows, targets]
    if np.any(~np.isfinite(y)) or np.any(y < 0):
        raise ValueError("Gumbel training scores must be finite nonnegative SW scores")
    query_counts = np.sum(leaders >= 0, axis=1)
    if query_weights is None:
        query_weights = _family_weights(families)
    else:
        query_weights = np.asarray(query_weights, dtype=float)
        if (query_weights.shape != families.shape
                or np.any(~np.isfinite(query_weights))
                or np.any(query_weights <= 0)):
            raise ValueError("Gumbel query weights must be positive and finite")
    weights = query_weights[rows] / query_counts[rows]
    weights /= weights.sum()
    qlog = np.log(query_lengths[rows])
    clog = np.log(cluster_lengths[cols])
    raw = np.column_stack((qlog, clog, np.abs(qlog - clog),
                           np.asarray(entropies)[rows]))
    if censored:
        shorter = np.minimum(query_lengths[rows], cluster_lengths[cols])
        ratio = np.maximum(query_lengths[rows], cluster_lengths[cols]) / shorter
        strata = 3 * np.digitize(shorter, [30, 60, 120]) + np.digitize(ratio, [1.5, 3])
        threshold = np.empty(len(y))
        fallback = _weighted_quantile(y, weights, .95)
        for stratum in np.unique(strata):
            mask = strata == stratum
            threshold[mask] = (_weighted_quantile(y[mask], weights[mask], .95)
                               if mask.sum() >= 200 else fallback)
        exact = (y > threshold).astype(float)
        observed = np.maximum(y, threshold)
    else:
        exact = np.ones(len(y))
        observed = y
    mean_y = np.average(y, weights=weights)
    beta0 = max(np.sqrt(np.average((y - mean_y) ** 2,
                                     weights=weights)) * math.sqrt(6) / math.pi, 1e-4)
    n_observations = len(y)
    n_tail_observations = int(exact.sum())
    if compress:
        # Query, cluster median length and censored score determine identical
        # likelihood terms. Summing their weights is an exact reduction.
        keys = np.rec.fromarrays([rows, cluster_lengths[cols], observed, exact],
                                 names="query,length,score,exact")
        _, first, inverse = np.unique(keys, return_index=True,
                                      return_inverse=True)
        weights = np.bincount(inverse, weights=weights)
        raw, observed, exact = raw[first], observed[first], exact[first]
    mean = np.average(raw, axis=0, weights=weights)
    scale = np.maximum(np.sqrt(np.average((raw - mean) ** 2,
                                          axis=0, weights=weights)), 1e-8)
    x = np.column_stack([np.ones(len(observed)), (raw - mean) / scale])
    initial = np.zeros((2, 5))
    initial[0, 0] = mean_y - .5772156649 * beta0
    initial[1, 0] = math.log(beta0)

    def objective(flat):
        coefficients = flat.reshape(2, 5)
        mu = x @ coefficients[0]
        linear = x @ coefficients[1]
        logbeta = np.clip(linear, -20, 10)
        invbeta = np.exp(-logbeta)
        z = (observed - mu) * invbeta
        expminus = np.exp(np.clip(-z, -700, 50))
        active = ((-z > -700) & (-z < 50)).astype(float)
        loss = float(np.sum(weights * (exact * (logbeta + z) + expminus)))
        grad_mu = x.T @ (weights * (expminus * active - exact) * invbeta)
        grad_beta = x.T @ (weights * (exact * (1-z) + z * expminus * active)
                           * ((linear > -20) & (linear < 10)))
        grad = np.r_[grad_mu, grad_beta]
        penalized = np.r_[1:5, 6:10]
        loss += 1e-3 * float(np.sum(flat[penalized] ** 2))
        grad[penalized] += 2e-3 * flat[penalized]
        violation = linear - logbeta
        loss += 100 * float(np.sum(weights * violation ** 2))
        grad[5:] += 200 * x.T @ (weights * violation)
        return loss, grad

    result = minimize(objective, initial.ravel(), method="L-BFGS-B", jac=True,
                      options={"maxiter": 800, "ftol": 1e-11, "gtol": 1e-7})
    loss, gradient = objective(result.x)
    if not np.isfinite(loss) or np.max(np.abs(gradient)) > 2e-3:
        raise RuntimeError(f"Gumbel fit did not converge: {result.message}; "
                           f"max gradient={np.max(np.abs(gradient)):.4g}")
    return {"coefficients_mu_logbeta": result.x.reshape(2, 5).T.tolist(),
            "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
            "censoring": "left-censored-95th-percentile" if censored else "none",
            "n_observations": int(n_observations), "n_queries": int(len(families)),
            "n_tail_observations": n_tail_observations,
            "n_compressed_observations": int(len(observed)),
            "max_abs_gradient": float(np.max(np.abs(gradient)))}


def _model_arrays(model):
    coefficients = np.asarray(model["coefficients_mu_logbeta"], dtype=float)
    mean = np.asarray(model["feature_mean"], dtype=float)
    scale = np.asarray(model["feature_scale"], dtype=float)
    lengths = np.asarray(model["cluster_median_lengths"], dtype=float)
    counts = np.asarray(model["cluster_multiplicities"], dtype=int)
    if (coefficients.shape != (5, 2) or mean.shape != (4,) or scale.shape != (4,)
            or lengths.ndim != 1 or counts.shape != lengths.shape or not len(lengths)
            or np.any(~np.isfinite(coefficients)) or np.any(~np.isfinite(mean))
            or np.any(~np.isfinite(scale)) or np.any(scale <= 0)
            or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0)
            or np.any(counts <= 0)):
        raise ValueError("invalid target-aware Gumbel parameters")
    return coefficients, mean, scale, lengths, counts


def initial_evalue(scores, query_length, entropy, model):
    """Sum each target cluster's Gumbel survival at the candidate raw SW score."""
    coefficients, mean, scale, lengths, counts = _model_arrays(model)
    if query_length <= 0 or not math.isfinite(entropy):
        raise ValueError("invalid query length or entropy")
    qlog = math.log(query_length)
    clog = np.log(lengths)
    raw = np.column_stack([np.full(len(lengths), qlog), clog,
                           np.abs(qlog - clog), np.full(len(lengths), entropy)])
    x = np.column_stack([np.ones(len(lengths)), (raw - mean) / scale])
    predicted = x @ coefficients
    mu = predicted[:, 0]
    beta = np.exp(np.clip(predicted[:, 1], -20, 10))
    values = np.asarray(scores, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("candidate scores must be finite nonnegative raw SW scores")
    flat = values.reshape(-1)
    out = np.empty(len(flat))
    for start in range(0, len(flat), 128):
        z = (flat[start:start + 128, None] - mu) / beta
        log_tail = np.empty_like(z)
        extreme = z > 16
        log_tail[extreme] = -z[extreme]
        log_tail[~extreme] = np.log(-np.expm1(-np.exp(
            np.clip(-z[~extreme], -700, 50))))
        out[start:start + 128] = np.exp(np.maximum(
            logsumexp(log_tail + np.log(counts), axis=1), -700))
    out = out.reshape(values.shape)
    return float(out) if out.ndim == 0 else out


def fit_empirical(evalue_rows, families, expansions, n_clusters, ceiling=100.0,
                  background_scaling="annotated-to-full N_D/B_q"):
    """Family-balanced cumulative unrelated-cluster count and low-tail slope."""
    families = np.asarray(families)
    expansions = np.asarray(expansions, dtype=float)
    if len(evalue_rows) != len(families) or len(families) != len(expansions):
        raise ValueError("E-value rows and query metadata differ")
    if n_clusters < 1 or np.any(~np.isfinite(expansions)) or np.any(expansions <= 0):
        raise ValueError("invalid cluster count or background expansion")
    query_weights = _family_weights(families) * expansions
    values, weights = [], []
    for row, weight in zip(evalue_rows, query_weights):
        arr = np.asarray(row, dtype=float)
        if np.any(~np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError("background E-values must be finite and nonnegative")
        selected = arr[(arr > 0) & (arr <= ceiling)]
        values.extend(selected.tolist())
        weights.extend([weight] * len(selected))
    if len(values) < 8:
        raise ValueError("too few supported background observations")
    values = np.asarray(values)
    weights = np.asarray(weights)
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    unique, first = np.unique(values, return_index=True)
    cumulative = np.cumsum(np.add.reduceat(weights, first))
    grid = np.geomspace(max(float(unique[0]), 1e-8), ceiling, 96)
    positions = np.searchsorted(unique, grid, side="right")
    observed = np.zeros(len(grid))
    present = positions > 0
    observed[present] = cumulative[positions[present] - 1]
    if present.sum() < 8:
        raise ValueError("too few supported thresholds for low-tail slope")
    slope, _ = np.polyfit(np.log(grid[present]), np.log(observed[present]), 1)
    if not np.isfinite(slope) or slope <= 0:
        raise ValueError("low-tail power slope must be positive")
    return {"scheme": "family-balanced-monotone", "raw_evalues": unique.tolist(),
            "observed_counts": cumulative.tolist(), "low_tail_power_b": float(slope),
            "upper_tail_rule": "anchored-power-cap",
            "fit_raw_evalue_min": float(unique[0]),
            "fit_raw_evalue_max": float(unique[-1]), "fit_raw_evalue_ceiling": float(ceiling),
            "fit_reported_evalue_max": float(cumulative[-1]),
            "n_clusters": int(n_clusters), "n_queries": int(len(families)),
            "n_families": int(len(np.unique(families))),
            "background_scaling": background_scaling}


def apply_empirical(evalues, calibration):
    """Interpolate observed counts and apply endpoint-anchored tails."""
    x = np.asarray(evalues, dtype=float)
    xp = np.asarray(calibration["raw_evalues"], dtype=float)
    yp = np.asarray(calibration["observed_counts"], dtype=float)
    b = float(calibration["low_tail_power_b"])
    n_clusters = int(calibration["n_clusters"])
    if (xp.ndim != 1 or yp.shape != xp.shape or len(xp) < 2
            or np.any(~np.isfinite(xp)) or np.any(~np.isfinite(yp))
            or np.any(xp <= 0) or np.any(np.diff(xp) <= 0)
            or np.any(yp <= 0) or np.any(np.diff(yp) < 0)
            or not math.isfinite(b) or b <= 0 or n_clusters < 1
            or yp[-1] > n_clusters * (1 + 1e-9)):
        raise ValueError("invalid empirical E-value calibration")
    if np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ValueError("initial E-values must be finite and nonnegative")
    y = np.interp(np.clip(x, xp[0], xp[-1]), xp, yp)
    if np.any(x < xp[0]):
        low = np.maximum(x, np.nextafter(0., 1.))
        extrapolated = np.exp(np.maximum(
            np.log(yp[0]) + b * (np.log(low) - np.log(xp[0])),
            np.log(np.nextafter(0., 1.))))
        y = np.where(x < xp[0], np.where(x == 0, 0., extrapolated), y)
    if calibration.get("upper_tail_rule") == "anchored-power-cap":
        upper = np.exp(np.minimum(
            np.log(yp[-1]) + b * (np.log(np.maximum(x, xp[-1])) - np.log(xp[-1])),
            np.log(n_clusters)))
        y = np.where(x > xp[-1], upper, y)
    elif calibration.get("upper_tail_rule") not in (None, "cluster-cap"):
        raise ValueError("invalid empirical upper-tail rule")
    else:  # Preserve the behaviour of parameters fitted before this change.
        y = np.where(x > xp[-1], float(n_clusters), y)
    y = np.minimum(y, n_clusters)
    return float(y) if np.ndim(y) == 0 else y


def validate_audit_metadata(params):
    """Check the audited range, including files with the old fixed-100 label."""
    status = params.get("validation_status", "")
    prefix = "audited-e0-le-"
    if not isinstance(status, str) or not status.startswith(prefix):
        raise ValueError("empirical E-values lack a completed structural audit")
    try:
        labelled = float(status[len(prefix):])
        ceiling = float(params.get("audit_raw_evalue_ceiling", labelled))
    except (TypeError, ValueError):
        raise ValueError("invalid empirical audit ceiling") from None
    if any(not math.isfinite(value) or value < 100 for value in (labelled, ceiling)):
        raise ValueError("empirical audit ceilings must be finite and at least 100")
    # Older fitters always wrote 100 in the label, even when the explicit
    # audit_raw_evalue_ceiling recorded a larger, completed audit.
    if labelled != ceiling and labelled != 100:
        raise ValueError("empirical audit label and ceiling disagree")
    calibration = params["empirical_calibration"]
    try:
        fit_ceiling = float(calibration.get("fit_raw_evalue_ceiling", ceiling))
        last_knot = float(calibration["raw_evalues"][-1])
        fit_max = float(calibration.get("fit_raw_evalue_max", last_knot))
    except (TypeError, ValueError):
        raise ValueError("invalid empirical calibration range") from None
    if (not math.isfinite(fit_ceiling) or not 0 < fit_ceiling <= ceiling
            or any(not math.isfinite(value) or not 0 < value <= fit_ceiling
                   for value in (last_knot, fit_max))):
        raise ValueError("empirical calibration range exceeds its audit or fitting ceiling")


def validate_parameters(params, n_clusters, target_fingerprint, context_fingerprint):
    if params.get("scheme") != SCHEME or params.get("score_definition") != "raw-sw":
        raise ValueError("unsupported experimental E-value scheme or score")
    if (params.get("target_database_fingerprint") != target_fingerprint
            or params.get("calibration_context_fingerprint") != context_fingerprint
            or int(params.get("n_clusters", -1)) != n_clusters):
        raise ValueError("experimental E-values belong to another database or lack audit")
    model = params["target_aware_gumbel"]
    _, _, _, _, multiplicities = _model_arrays(model)
    if int(multiplicities.sum()) != n_clusters:
        raise ValueError("Gumbel cluster distribution does not match database")
    calibration = params["empirical_calibration"]
    if int(calibration.get("n_clusters", -1)) != n_clusters:
        raise ValueError("empirical map cluster count does not match database")
    apply_empirical(np.array([0.0, 1.0]), calibration)
    validate_audit_metadata(params)
