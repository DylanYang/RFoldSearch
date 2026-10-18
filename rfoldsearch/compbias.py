"""Position-specific token scores for composition-biased query regions."""
import numpy as np


WINDOW = 20


def background_frequencies(token_lists, n_codes):
    counts = np.zeros(n_codes, dtype=np.float64)
    for tokens in token_lists:
        counts += np.bincount(np.asarray(tokens, dtype=np.int64), minlength=n_codes)
    if counts.sum() == 0:
        raise ValueError("cannot score against an empty database")
    return counts / counts.sum()


def query_profile(query, matrix, background):
    """Return position IDs and window-20 profile for Smith--Waterman."""
    query = np.asarray(query, dtype=np.int64)
    matrix = np.asarray(matrix, dtype=np.float32)
    if not len(query):
        raise ValueError("query tokens must be nonempty")
    if np.any(query < 0) or np.any(query >= matrix.shape[0]):
        raise ValueError("query token is outside the substitution matrix")
    background = np.asarray(background, dtype=np.float64)
    if (background.shape != (matrix.shape[0],)
            or np.any(~np.isfinite(background)) or np.any(background < 0)
            or not np.isclose(background.sum(), 1)):
        raise ValueError("invalid database token frequencies")
    rows = matrix[query]
    if len(query) == 1:
        return np.arange(1, dtype=np.int64), np.ascontiguousarray(rows)
    prefix = np.vstack((np.zeros(matrix.shape[1]),
                        np.cumsum(rows, axis=0, dtype=np.float64)))
    starts = np.maximum(np.arange(len(query)) - WINDOW, 0)
    ends = np.minimum(np.arange(len(query)) + WINDOW + 1, len(query))
    local = (prefix[ends] - prefix[starts] - rows) / (ends - starts - 1)[:, None]
    expected = background @ matrix.astype(np.float64)
    profile = rows - local + expected[None, :]
    return np.arange(len(query), dtype=np.int64), np.ascontiguousarray(profile,
                                                                     dtype=np.float32)
