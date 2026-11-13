"""Exhaustively rescore a newly encoded RNA database (no prior score cache)."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from rfoldsearch.database import load_database
from rfoldsearch.search import GAP_EXTEND, GAP_OPEN, load_matrix
from rfoldsearch.smith_waterman import smith_waterman_numba


_WORKER = None


def init_worker(db_path: str, matrix_path: str, background: np.ndarray) -> None:
    global _WORKER
    _, _, token_lists, _, _, _ = load_database(db_path)
    _WORKER = (token_lists, load_matrix(matrix_path), background)


def score_query(index: int) -> tuple[np.ndarray, np.ndarray]:
    if _WORKER is None:
        raise RuntimeError("score worker was not initialised")
    token_lists, matrix, background = _WORKER
    query = np.asarray(token_lists[index], dtype=np.int64)
    positions = np.arange(len(query), dtype=np.int64)
    profile = corrected_profile(query, matrix, background)
    raw = np.empty(len(token_lists), dtype=np.float32)
    corrected = np.empty(len(token_lists), dtype=np.float32)
    for col, target in enumerate(token_lists):
        raw[col] = smith_waterman_numba(query, target, matrix, GAP_OPEN, GAP_EXTEND)
        corrected[col] = smith_waterman_numba(
            positions, target, profile, GAP_OPEN, GAP_EXTEND)
    return raw, corrected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def corrected_profile(query: np.ndarray, matrix: np.ndarray,
                      background: np.ndarray, window: int = 20) -> np.ndarray:
    rows = matrix[query]
    if len(query) < 2:
        return np.ascontiguousarray(rows, dtype=np.float32)
    prefix = np.vstack([np.zeros(matrix.shape[1]),
                        np.cumsum(rows, axis=0, dtype=np.float64)])
    positions = np.arange(len(query))
    start = np.maximum(positions - window, 0)
    end = np.minimum(positions + window + 1, len(query))
    local = (prefix[end] - prefix[start] - rows) / (end - start - 1)[:, None]
    expected = background @ matrix.astype(np.float64)
    return np.ascontiguousarray(rows - local + expected, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke test only; omit for the complete comparison")
    args = parser.parse_args()
    _, _, tokens, ids, n_codes, _ = load_database(str(args.db))
    labels = {}
    for line in args.labels.read_text(encoding="utf-8").splitlines()[1:]:
        chain, family, clan = (line.split("\t") + ["", ""])[:3]
        labels[chain] = (family, clan)
    if set(ids) != set(labels):
        raise ValueError("encoded database IDs and Rfam annotation IDs differ")
    annotated = [i for i, chain in enumerate(ids) if labels[chain][0]]
    unannotated = [i for i, chain in enumerate(ids) if not labels[chain][0]]
    def entropy(index: int) -> float:
        frequencies = np.bincount(tokens[index], minlength=n_codes).astype(float)
        probabilities = frequencies[frequencies > 0] / frequencies.sum()
        return float(-(probabilities * np.log(probabilities)).sum())
    short = sorted(unannotated, key=lambda i: (len(tokens[i]), ids[i]))[:38]
    low_entropy = sorted(set(unannotated) - set(short),
                         key=lambda i: (entropy(i), ids[i]))[:37]
    queries = np.array(annotated + short + low_entropy, dtype=np.int32)
    if args.limit is not None:
        queries = queries[:args.limit]
    matrix = load_matrix(str(args.matrix))
    if matrix.shape != (n_codes, n_codes):
        raise ValueError("matrix alphabet size differs from the encoded database")
    counts = np.zeros(n_codes, dtype=np.float64)
    for tok in tokens:
        counts += np.bincount(tok, minlength=n_codes)
    background = counts / counts.sum()
    args.out.mkdir(parents=True, exist_ok=True)
    raw = np.lib.format.open_memmap(args.out / "raw_sw.npy", mode="w+",
                                    dtype=np.float32, shape=(len(queries), len(tokens)))
    corrected = np.lib.format.open_memmap(args.out / "compbias20_sw.npy", mode="w+",
                                          dtype=np.float32, shape=raw.shape)
    if args.workers == 1:
        init_worker(str(args.db), str(args.matrix), background)
        rows = map(score_query, queries)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                   initargs=(str(args.db), str(args.matrix), background))
        rows = pool.map(score_query, queries)
    try:
        for row, (raw_scores, corrected_scores) in enumerate(rows):
            raw[row] = raw_scores
            corrected[row] = corrected_scores
            if (row + 1) % 10 == 0 or row + 1 == len(queries):
                raw.flush()
                corrected.flush()
                print(f"scored {row + 1}/{len(queries)} queries", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    if not (np.isfinite(raw).all() and np.isfinite(corrected).all()):
        raise ValueError("non-finite alignment score")
    self_raw = np.empty(len(tokens), dtype=np.float32)
    self_corrected = np.empty(len(tokens), dtype=np.float32)
    for index, tok in enumerate(tokens):
        query = np.asarray(tok, dtype=np.int64)
        self_raw[index] = smith_waterman_numba(
            query, query, matrix, GAP_OPEN, GAP_EXTEND)
        profile = corrected_profile(query, matrix, background)
        self_corrected[index] = smith_waterman_numba(
            np.arange(len(query), dtype=np.int64), query, profile,
            GAP_OPEN, GAP_EXTEND)
    np.save(args.out / "self_raw_sw.npy", self_raw)
    np.save(args.out / "self_compbias20_sw.npy", self_corrected)
    metadata = {"database_sha256": sha256(args.db),
                "labels_sha256": sha256(args.labels),
                "matrix_sha256": sha256(args.matrix),
                "n_targets": len(tokens), "n_queries": len(queries),
                "n_annotated_queries": min(len(annotated), len(queries)),
                "auxiliary_selection": "38 shortest and 37 lowest-entropy unannotated chains",
                "query_ids": [ids[i] for i in queries],
                "score_modes": ["raw SW", "window-20 composition-adjusted raw SW"],
                "gap_open": GAP_OPEN, "gap_extend": GAP_EXTEND}
    (args.out / "score_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
