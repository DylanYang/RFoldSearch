"""Length-binned power calibration, the pure power tail, and composition-bias E-values."""
import math

import numpy as np
import pytest
import torch

from rfoldsearch import evalue as EV
from rfoldsearch.compbias import background_frequencies, query_profile
from rfoldsearch.database import (calibration_context_fingerprint,
                                  database_fingerprint)
from rfoldsearch.search import GAP_EXTEND, GAP_OPEN, Searcher, load_matrix
from rfoldsearch.smith_waterman import smith_waterman_numba
from helpers import synthetic_evalue_params

BINS = [{"log_query_length": math.log(30), "a": 2.0, "b": 1.0},
        {"log_query_length": math.log(120), "a": 0.5, "b": 1.5}]


def _power(**extra):
    return {"a": 1.0, "b": 1.0, "fit_raw_evalue_max": 1000.0, "n_clusters": 500,
            "tail_rule": EV.TAIL_RULE_POWER, **extra}


def test_length_bins_interpolate_in_log_length_and_clamp():
    power = _power(length_bins=BINS)
    assert EV.power_terms(power, 30) == pytest.approx((2.0, 1.0))
    assert EV.power_terms(power, 10) == pytest.approx((2.0, 1.0))
    assert EV.power_terms(power, 400) == pytest.approx((0.5, 1.5))
    a, b = EV.power_terms(power, 60)                 # halfway in log length
    assert a == pytest.approx(1.0) and b == pytest.approx(1.25)
    with pytest.raises(ValueError, match="query length"):
        EV.apply_power_calibration(1.0, power)


def test_pure_power_tail_is_not_proportional_and_is_capped():
    power = _power(a=0.5, b=1.5)
    raw = np.array([0.0, 10.0, 2000.0, 1e6])
    assert EV.apply_power_calibration(raw, power) == pytest.approx(
        np.minimum(0.5 * raw ** 1.5, 500))


def test_without_bins_old_parameters_are_unchanged():
    power = {**_power(a=2.0, b=1.2), "tail_rule": EV.TAIL_RULE,
             "fit_raw_evalue_max": 100.0}
    assert EV.apply_power_calibration(50.0, power) == pytest.approx(2.0 * 50 ** 1.2)
    assert EV.apply_power_calibration(50.0, power, 30) == pytest.approx(2.0 * 50 ** 1.2)


@pytest.mark.parametrize("value,ok", [(False, True), (None, True), ({"window": 20}, True),
                                      ({"window": 0}, False), ({"window": True}, False),
                                      ({"size": 20}, False), (20, False)])
def test_composition_bias_setting_is_validated(value, ok):
    if ok:
        EV.composition_bias_window({"composition_bias": value})
    else:
        with pytest.raises(ValueError):
            EV.composition_bias_window({"composition_bias": value})


def test_context_fingerprint_binds_window_and_keeps_old_digest():
    matrix = np.eye(4, dtype=np.float32)
    plain = calibration_context_fingerprint("db", matrix, -2.0, -0.5)
    assert calibration_context_fingerprint("db", matrix, -2.0, -0.5, 0) == plain
    assert calibration_context_fingerprint("db", matrix, -2.0, -0.5, 20) != plain
    assert calibration_context_fingerprint("db", matrix, -2.0, -0.5, 20) != (
        calibration_context_fingerprint("db", matrix, -2.0, -0.5, 10))


def _corrected_db(tmp_path, window):
    rng = np.random.default_rng(0)
    entries = [{"id": f"c{i}", "tokens": rng.integers(0, 6, 40).tolist(), "cluster": f"k{i}"}
               for i in range(6)]
    params = synthetic_evalue_params(entries)
    params["power_calibration"].update(tail_rule=EV.TAIL_RULE_POWER,
                                       length_bins=BINS, fit_raw_evalue_max=1000.0)
    if window:
        params["composition_bias"] = {"window": window}
        clusters = [e["cluster"] for e in entries]
        params["calibration_context_fingerprint"] = calibration_context_fingerprint(
            database_fingerprint(entries, clusters, 80), load_matrix(),
            GAP_OPEN, GAP_EXTEND, window)
    db = tmp_path / f"db{window}.pt"
    torch.save({"n_codes": 80, "entries": entries, "evalue_params": params}, db)
    return entries, params, db


def test_corrected_scores_give_evalues_but_not_ranking(tmp_path):
    entries, params, db = _corrected_db(tmp_path, 5)
    _, _, plain_db = _corrected_db(tmp_path, 0)
    hits = Searcher(str(db)).search_id("c0", rerank="sw", top_k=5)
    plain = Searcher(str(plain_db)).search_id("c0", rerank="sw", top_k=5)
    assert [h["target"] for h in hits] == [h["target"] for h in plain]
    assert [h["sw_score"] for h in hits] == [h["sw_score"] for h in plain]
    tokens = [np.asarray(e["tokens"]) for e in entries]
    positions, profile = query_profile(tokens[0], load_matrix(),
                                       background_frequencies(tokens, 80), 5)
    for hit in hits:
        target = tokens[int(hit["target"][1:])]
        corrected = smith_waterman_numba(positions, target, profile, GAP_OPEN, GAP_EXTEND)
        assert hit["evalue_sw_score"] == pytest.approx(corrected)
        raw = EV.target_aware_evalue(params, tokens[0], corrected, len(target))
        assert hit["evalue"] == pytest.approx(EV.apply_power_calibration(
            raw, params["power_calibration"], 40))


def test_window_mismatch_is_rejected(tmp_path):
    entries, params, _ = _corrected_db(tmp_path, 5)
    params["composition_bias"] = {"window": 6}       # fingerprint was made for 5
    db = tmp_path / "bad.pt"
    torch.save({"n_codes": 80, "entries": entries, "evalue_params": params}, db)
    with pytest.raises(ValueError, match="scoring context"):
        Searcher(str(db))


def _fit_module():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "fit_target_aware_evalue_params.py"
    spec = importlib.util.spec_from_file_location("fit_ta", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fitter_bins_power_by_query_length_and_merges_sparse_bins():
    F = _fit_module()
    rng = np.random.default_rng(1)
    rows, initial = [], []
    for q in range(80):
        length = 20 + q          # four length groups of 20 queries
        scale = 3.0 if q < 40 else 0.3
        for e in rng.uniform(0.01, 50, 30):
            rows.append({"query_id": f"q{q}", "role": "power", "query_length": float(length),
                         "query_entropy": 2.0, "family": frozenset([f"f{q}"]),
                         "clan": frozenset(), "query_weight": 1.0, "expansion": 1.0})
            initial.append(e * scale)
    idx = list(range(len(rows)))
    fit = F.fit_power_length_bins(rows, idx, np.array(initial), 500, 1000.0, 64,
                                  EV.TAIL_RULE_POWER, 4)
    bins = fit["length_bins"]
    assert len(bins) == 4 and fit["tail_rule"] == EV.TAIL_RULE_POWER
    assert [b["n_queries"] for b in bins] == [20] * 4
    assert all(x["log_query_length"] < y["log_query_length"] for x, y in zip(bins, bins[1:]))
    # short queries' E0 is inflated, so their calibrated E must come out smaller
    assert EV.apply_power_calibration(10.0, {**fit, "fit_raw_evalue_max": 1000.0}, 25) < (
        EV.apply_power_calibration(10.0, {**fit, "fit_raw_evalue_max": 1000.0}, 95))
    fewer = F.fit_power_length_bins(rows[:30 * 50], idx[:30 * 50], np.array(initial[:30 * 50]),
                                    500, 1000.0, 64, EV.TAIL_RULE_POWER, 4)
    assert len(fewer["length_bins"]) == 2          # 50 queries: at most 2 bins of >= 20
    single = F.fit_power_length_bins(rows, idx, np.array(initial), 500, 1000.0, 64,
                                     EV.TAIL_RULE_POWER, 1)
    assert "length_bins" not in single


def test_raw_score_model_skips_length_normalization():
    entries = [{"id": "a", "tokens": [0] * 10, "cluster": "c0"},
               {"id": "b", "tokens": [0] * 100, "cluster": "c1"}]
    params = synthetic_evalue_params(entries)
    raw = {**params, "target_aware_gumbel": {**params["target_aware_gumbel"],
                                             "score_normalization": "sw"}}
    query = np.array([0] * 20)
    assert EV.target_aware_evalue(raw, query, 7.0, 100) == pytest.approx(
        EV.target_aware_initial_evalue(raw, query, 7.0))
    assert EV.target_aware_evalue(params, query, 7.0, 100) == pytest.approx(
        EV.target_aware_initial_evalue(params, query, 7.0 / 20))
    with pytest.raises(ValueError, match="normalization"):
        EV.target_aware_evalue({**params, "target_aware_gumbel": {
            **params["target_aware_gumbel"], "score_normalization": "sw/len"}}, query, 7.0, 100)
