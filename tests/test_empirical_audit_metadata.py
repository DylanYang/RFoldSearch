"""Reject unaudited ranges without invalidating historical parameter files."""
import pytest

from rfoldsearch.experimental_evalue import validate_audit_metadata


@pytest.mark.parametrize("status,ceiling,fit_ceiling,fit_max,knots", [
    ("audited-e0-le-100", None, 100, 99, [.1, 99]),
    ("audited-e0-le-400", 400, 400, 399, [.1, 399]),
    ("audited-e0-le-100", 400, 400, 399, [.1, 399]),
    ("audited-e0-le-400.125", 400.125, 400.125, 400.1, [.1, 400.1]),
    ("audited-e0-le-400", 400, 100, 99, [.1, 99]),
])
def test_supported_audit_ranges(status, ceiling, fit_ceiling, fit_max, knots):
    params = {"validation_status": status, "empirical_calibration": {
        "fit_raw_evalue_ceiling": fit_ceiling, "fit_raw_evalue_max": fit_max,
        "raw_evalues": knots}}
    if ceiling is not None:
        params["audit_raw_evalue_ceiling"] = ceiling
    validate_audit_metadata(params)


@pytest.mark.parametrize("changes", [
    {"validation_status": "structural-audit-not-recorded"},
    {"validation_status": "audited-e0-le-nan"},
    {"validation_status": "audited-e0-le-400junk"},
    {"validation_status": None},
    {"validation_status": "audited-e0-le-500"},
    {"audit_raw_evalue_ceiling": 100},
    {"audit_raw_evalue_ceiling": 500},
    {"audit_raw_evalue_ceiling": float("inf")},
    {"audit_raw_evalue_ceiling": 99},
    {"audit_raw_evalue_ceiling": None},
])
def test_invalid_audit_ranges(changes):
    params = {"validation_status": "audited-e0-le-400",
              "audit_raw_evalue_ceiling": 400,
              "empirical_calibration": {"fit_raw_evalue_ceiling": 400,
                  "fit_raw_evalue_max": 399, "raw_evalues": [.1, 399]}}
    params.update(changes)
    with pytest.raises(ValueError):
        validate_audit_metadata(params)


@pytest.mark.parametrize("changes", [
    {"fit_raw_evalue_ceiling": 500},
    {"fit_raw_evalue_ceiling": float("nan")},
    {"fit_raw_evalue_max": 401},
    {"fit_raw_evalue_max": float("inf")},
    {"raw_evalues": [.1, 401]},
])
def test_calibration_cannot_exceed_audited_range(changes):
    calibration = {"fit_raw_evalue_ceiling": 400,
                   "fit_raw_evalue_max": 399, "raw_evalues": [.1, 399]}
    calibration.update(changes)
    with pytest.raises(ValueError, match="calibration range"):
        validate_audit_metadata({"validation_status": "audited-e0-le-400",
            "audit_raw_evalue_ceiling": 400, "empirical_calibration": calibration})
