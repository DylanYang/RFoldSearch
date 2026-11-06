"""Deployment checks for predicted-database composition-adjusted E-values."""
import numpy as np

from . import compbias
from . import experimental_evalue as core


SCHEME = "target-aware-compbias20-raw-sw-empirical-v1"


def validate_parameters(params, n_clusters, target_fingerprint,
                        context_fingerprint, tokens):
    if (params.get("scheme") != SCHEME
            or params.get("score_definition") != "compbias20-raw-sw"
            or params.get("composition_window") != compbias.WINDOW
            or params.get("target_database_fingerprint") != target_fingerprint
            or params.get("calibration_context_fingerprint") != context_fingerprint
            or int(params.get("n_clusters", -1)) != n_clusters):
        raise ValueError("predicted E-values do not match this database and scoring context")
    model = params["target_aware_gumbel"]
    _, _, _, _, multiplicities = core._model_arrays(model)
    if int(multiplicities.sum()) != n_clusters:
        raise ValueError("predicted Gumbel cluster count does not match database")
    calibration = params["empirical_calibration"]
    if (int(calibration.get("n_clusters", -1)) != n_clusters
            or calibration.get("background_scaling") != "complete-eligible-clusters"):
        raise ValueError("predicted empirical calibration needs the full eligible background")
    core.apply_empirical(np.array([0.0, 1.0]), calibration)
    core.validate_audit_metadata(params)
    fitted = np.asarray(params["composition_background"], dtype=float)
    if fitted.ndim != 1 or not len(fitted):
        raise ValueError("invalid predicted composition background")
    observed = compbias.background_frequencies(tokens, len(fitted))
    if fitted.shape != observed.shape or not np.allclose(fitted, observed, atol=1e-10):
        raise ValueError("composition background does not match target tokens")
