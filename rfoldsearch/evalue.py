"""Database-specific target-aware Gumbel E-values for normalized token SW scores."""

from __future__ import annotations

import math

import numpy as np


TAIL_RULE = "proportional above fit_max; capped at cluster count"
# Pure power law over the whole E0 range. Fit it to an E0 range that reaches
# well past the significance region (for example E0 <= 1000) so the curve is
# not extrapolated where short queries' background E-values lie.
TAIL_RULE_POWER = "power law throughout; capped at cluster count"
TAIL_RULES = (TAIL_RULE, TAIL_RULE_POWER)
FEATURE_ORDER = [
    "intercept", "z_log_query_length", "z_log_cluster_median_length",
    "z_log_length_difference", "z_token_entropy",
]
FEATURE_NAMES = [
    "log_query_length", "log_cluster_median_length",
    "log_length_difference", "token_entropy",
]


def _token_entropy(tokens):
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.ndim != 1 or not len(tokens) or np.any(tokens < 0):
        raise ValueError("query tokens must be a nonempty 1D array of nonnegative IDs")
    counts = np.bincount(tokens)
    probabilities = counts[counts > 0] / len(tokens)
    return float(-(probabilities * np.log2(probabilities)).sum())


def _target_aware_model(params: dict):
    model = params.get("target_aware_gumbel")
    if not isinstance(model, dict) or model.get("distribution") != "Gumbel":
        raise ValueError("target-aware E-values require a Gumbel model")
    if model.get("score_normalization") != "sw/min(query_length,target_length)":
        raise ValueError("unsupported target-aware score normalization")
    if model.get("censoring") not in {"left-censored", "none"}:
        raise ValueError("target-aware model must record its fitted censoring mode")
    if model.get("feature_order") != FEATURE_ORDER:
        raise ValueError("unsupported target-aware feature order")
    centre = model.get("centre")
    if not isinstance(centre, dict):
        raise ValueError("target-aware model needs feature centres")
    for name in FEATURE_NAMES:
        values = centre.get(name)
        if not isinstance(values, dict):
            raise ValueError(f"target-aware model needs centre for {name}")
        mean = float(values.get("mean", float("nan")))
        scale = float(values.get("scale", float("nan")))
        if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid target-aware centre for {name}")
    coefficients = np.asarray(model.get("coefficients_mu_logbeta"), dtype=np.float64)
    if coefficients.shape != (5, 2) or not np.all(np.isfinite(coefficients)):
        raise ValueError("target-aware coefficients must be finite 5x2")
    lengths = np.asarray(model.get("cluster_median_lengths"), dtype=np.float64)
    multiplicities = np.asarray(model.get("cluster_multiplicities"), dtype=np.int64)
    if (lengths.ndim != 1 or multiplicities.ndim != 1 or not len(lengths)
            or len(lengths) != len(multiplicities)
            or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0)
            or np.any(multiplicities <= 0)):
        raise ValueError("invalid target-cluster length distribution")
    if int(multiplicities.sum()) != int(params.get("n_clusters", -1)):
        raise ValueError("target-cluster distribution does not match n_clusters")
    return centre, coefficients, lengths, multiplicities


def power_terms(calibration: dict, query_length=None):
    """-> (a, b) for a query of this length.

    With ``length_bins`` the power law is fitted separately per query-length
    bin; log a and b are interpolated linearly in log query length between
    the bin centres and held constant beyond the outermost centres.
    """
    bins = calibration.get("length_bins")
    if not bins:
        return float(calibration["a"]), float(calibration["b"])
    if query_length is None:
        raise ValueError("length-binned power calibration needs the query length")
    if not query_length > 0:
        raise ValueError("query length must be positive")
    centres = [float(b["log_query_length"]) for b in bins]
    x = math.log(float(query_length))
    log_a = float(np.interp(x, centres, [math.log(float(b["a"])) for b in bins]))
    b = float(np.interp(x, centres, [float(b["b"]) for b in bins]))
    return math.exp(log_a), b


def apply_power_calibration(raw_evalue, calibration: dict, query_length=None):
    """Map E0 through the validated power fit, its tail rule, and the cluster cap."""
    values = np.asarray(raw_evalue, dtype=np.float64)
    a, b = power_terms(calibration, query_length)
    ceiling = float(calibration["fit_raw_evalue_max"])
    n_clusters = int(calibration["n_clusters"])
    if (not all(map(math.isfinite, (a, b, ceiling))) or a <= 0 or b <= 0
            or ceiling <= 0 or n_clusters < 1):
        raise ValueError("power calibration requires finite positive parameters")
    rule = calibration.get("tail_rule")
    if rule not in TAIL_RULES:
        raise ValueError("unsupported power extrapolation rule")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("raw E-values must be finite and non-negative")
    if rule == TAIL_RULE:
        output = a * np.power(np.minimum(values, ceiling), b)
        output *= np.maximum(1.0, values / ceiling)
    else:
        output = a * np.power(values, b)
    output = np.minimum(output, n_clusters)
    return float(output) if output.ndim == 0 else output


def composition_bias_window(params: dict) -> int:
    """Half-width of the query composition-bias correction the fit used (0: none)."""
    value = params.get("composition_bias", False)
    if value is False or value is None:
        return 0
    if not isinstance(value, dict) or set(value) != {"window"}:
        raise ValueError("composition_bias must be false or {'window': n}")
    window = value["window"]
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("composition-bias window must be a positive integer")
    return window


def _validate_length_bins(power: dict) -> None:
    bins = power.get("length_bins")
    if bins is None:
        return
    if not isinstance(bins, list) or not bins:
        raise ValueError("length_bins must be a nonempty list")
    centres = []
    for entry in bins:
        try:
            centre, a, b = (float(entry[k]) for k in ("log_query_length", "a", "b"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each length bin needs log_query_length, a and b") from exc
        if not all(map(math.isfinite, (centre, a, b))) or a <= 0 or b <= 0:
            raise ValueError("length-bin power parameters must be finite and positive")
        centres.append(centre)
    if any(y <= x for x, y in zip(centres, centres[1:])):
        raise ValueError("length-bin centres must increase strictly")


def validate_target_aware_params(params: dict, n_clusters: int,
                                 target_fingerprint: str,
                                 score_fingerprint: str | None = None,
                                 cluster_median_lengths=None) -> None:
    """Reject legacy or unverified assets before attaching or searching."""
    if any(key in params for key in ("query_parameter_model", "length_table",
                                     "empirical_calibration")):
        raise ValueError("E-value asset contains obsolete calibration parameters")
    model = params.get("target_aware_gumbel")
    if not isinstance(model, dict):
        raise ValueError("only target-aware E-value parameters are supported")
    if params.get("scheme") != f"target-aware-{model.get('censoring')}-gumbel-power":
        raise ValueError("target-aware E-value scheme does not match the fitted model")
    if (params.get("validation_status") != "background-audit-complete"
            or params.get("target_database_fingerprint") != target_fingerprint
            or not params.get("calibration_context_fingerprint")
            or int(params.get("n_clusters", -1)) != n_clusters):
        raise ValueError("E-value parameters are unvalidated or fitted for a different database")
    if (score_fingerprint is not None
            and params["calibration_context_fingerprint"] != score_fingerprint):
        raise ValueError("E-value parameters were fitted for a different scoring context")
    _, _, lengths, multiplicities = _target_aware_model(params)
    if cluster_median_lengths is not None:
        actual_lengths, actual_counts = np.unique(
            np.asarray(cluster_median_lengths, dtype=np.float64), return_counts=True)
        if (not np.array_equal(lengths, actual_lengths)
                or not np.array_equal(multiplicities, actual_counts)):
            raise ValueError("target-cluster lengths do not match the database")
    power = params.get("power_calibration")
    if (not isinstance(power, dict)
            or power.get("validation_status") != "background-audit-complete"
            or int(power.get("n_clusters", -1)) != n_clusters):
        raise ValueError("target-aware power calibration is not validated for this database")
    _validate_length_bins(power)
    composition_bias_window(params)
    minimum = float(power.get("fit_raw_evalue_min", float("nan")))
    maximum = float(power.get("fit_raw_evalue_max", float("nan")))
    if not math.isfinite(minimum) or not 0 < minimum <= maximum:
        raise ValueError("invalid power calibration fit range")
    for bin_ in power.get("length_bins") or [{"log_query_length": 0.0}]:
        apply_power_calibration(0.0, power, math.exp(float(bin_["log_query_length"])))


def target_aware_initial_evalue(params: dict, query_tokens, normalized_score):
    """Expected count of target clusters whose maximum normalized SW exceeds s."""
    centre, coefficients, cluster_lengths, multiplicities = _target_aware_model(params)
    query = np.asarray(query_tokens, dtype=np.int64)
    qlength = len(query)
    entropy = _token_entropy(query)
    score = np.asarray(normalized_score, dtype=np.float64)
    if np.any(~np.isfinite(score)):
        raise ValueError("normalized Smith-Waterman score must be finite")
    qlog = math.log(qlength)
    clog = np.log(cluster_lengths)
    raw = np.column_stack([
        np.full(len(cluster_lengths), qlog), clog,
        np.abs(qlog - clog), np.full(len(cluster_lengths), entropy),
    ])
    x = np.ones((len(cluster_lengths), 5), dtype=np.float64)
    for column, name in enumerate(FEATURE_NAMES, 1):
        x[:, column] = ((raw[:, column - 1] - centre[name]["mean"])
                        / centre[name]["scale"])
    mu, log_beta = x @ coefficients[:, 0], x @ coefficients[:, 1]
    beta = np.exp(np.clip(log_beta, -20.0, 10.0))
    z = np.clip(-(score[..., None] - mu) / beta, -745.0, 50.0)
    tail = -np.expm1(-np.exp(z))
    result = np.sum(tail * multiplicities, axis=-1)
    return float(result) if result.ndim == 0 else result


def target_aware_evalue(params: dict, query_tokens, sw_score, target_length: int):
    """Use actual hit length in SW/min(Lq,Lt), then count null clusters."""
    query_length = len(np.asarray(query_tokens))
    if query_length < 1 or int(target_length) < 1:
        raise ValueError("query and target lengths must be positive")
    score = np.asarray(sw_score, dtype=np.float64) / min(query_length, int(target_length))
    return target_aware_initial_evalue(params, query_tokens, score)
