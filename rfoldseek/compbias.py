"""Composition-bias correction of the query, as a position-specific score table.

Predicted structures are often long runs of a few helix states. Two such
chains align for a high token score that says only that both are helical, and
a query of that kind cannot tell its own structure from hundreds of unrelated
ones. As in MMseqs2 and Foldseek, each query position i is scored against
target state a with

    P[i, a] = S(q_i, a) - (mean_{j in window(i), j != i} S(q_j, a) - E_bg[S(., a)]),

so a match the query's own neighbourhood would make anyway earns little,
while a state that stands out from its surroundings keeps its full score.

The aligners index scores as ``sub[query[i], target[j]]``; passing the
positions ``0..L-1`` as the query and ``P`` as the matrix makes them use the
profile unchanged. With ``window=0`` the profile is ``S[query]`` and the
scores are exactly the uncorrected ones.
"""
import numpy as np

DEFAULT_WINDOW = 20       # half-width, in tokens, when the correction is enabled


def background_frequencies(token_lists, n_codes: int) -> np.ndarray:
    """State frequencies of a database, the reference composition."""
    counts = np.zeros(int(n_codes), dtype=np.float64)
    for tokens in token_lists:
        counts += np.bincount(np.asarray(tokens, dtype=np.int64), minlength=int(n_codes))
    if counts.sum() <= 0:
        raise ValueError("cannot take a background composition of an empty database")
    return counts / counts.sum()


def query_profile(query, sub, background=None, window: int = 0):
    """-> (positions, profile) to pass to the aligners in place of (query, sub)."""
    query = np.asarray(query, dtype=np.int64)
    sub = np.asarray(sub, dtype=np.float32)
    positions = np.arange(len(query), dtype=np.int64)
    rows = sub[query]
    window = int(window)
    if window < 0:
        raise ValueError("composition-bias window must be non-negative")
    if window == 0 or len(query) < 2:
        return positions, np.ascontiguousarray(rows)
    if background is None:
        raise ValueError("composition-bias correction needs a background composition")
    expected = np.asarray(background, dtype=np.float64) @ sub.astype(np.float64)
    total = np.vstack([np.zeros(sub.shape[1]), np.cumsum(rows, axis=0, dtype=np.float64)])
    starts = np.maximum(np.arange(len(query)) - window, 0)
    ends = np.minimum(np.arange(len(query)) + window + 1, len(query))
    local = (total[ends] - total[starts] - rows) / (ends - starts - 1)[:, None]
    profile = rows - (local - expected[None, :])
    return positions, np.ascontiguousarray(profile, dtype=np.float32)
