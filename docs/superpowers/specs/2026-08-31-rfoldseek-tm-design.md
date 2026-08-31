# RFoldseek-TM: token prefilter -> US-align TM-score reranking

Date: 2026-08-31
Status: approved design, not yet implemented

## Motivation

RFoldseek ranks hits by an affine-gap Smith-Waterman score over 3Dr tokens.
The token alignment is fast but it is a proxy for structural similarity, not a
measurement of it. RFoldseek-TM keeps the token prefilter — which is what makes
the search fast — and replaces only the reranking stage with a real structural
superposition, scoring each candidate by US-align's TM-score.

This is the Foldseek-style coordinate-reranking variant used in the paper's
remote-homologue analysis. `RFoldseek-paper-code`'s
`scripts/02_benchmark/eval_remote_rfoldseek_tm.py` defines it as: the token
prefilter supplies a candidate pool, then US-align reranks only those candidates
by the mean of its two length-normalised TM-scores. That script only *consumes*
precomputed US-align score caches; no code in either repository actually runs
US-align. This design supplies that missing piece.

Foldseek does the same thing with `--alignment-type 1`: 3Di/AA as a prefilter,
then global TM-align on the survivors. It differs in one respect — Foldseek
sorts by TM normalised by query length, whereas RFoldseek-TM uses the mean of
the two normalisations, matching the paper.

## Decisions taken

1. **TM-score comes from the external `USalign` binary**, invoked as a
   subprocess, not from a vendored C++ extension and not from a
   reimplementation.

   US-align is BSD/MIT-style licensed and could legally be vendored, but doing
   so would mean adding a wheel-building pipeline (the package is pure Python
   today, so `pip install .` needs no compiler) and refactoring US-align from a
   `main()`-plus-globals program into a re-entrant library. The subprocess keeps
   packaging untouched and makes the numbers identical to the paper's by
   construction. Vendoring stays open as a later change if "works straight after
   installation" becomes a requirement for TM mode.

2. **Coordinates are written from the token database, not read back from the
   original structure files.**

   US-align uses C3' as the representative atom for RNA, the way it uses CA for
   proteins. Database entries already store `coords` of shape `(L, 5, 3)` whose
   first channel is C3' (`tokenizer.py:100`), so a synthesised C3'-only PDB
   yields the same TM-score as running US-align on the original file — the other
   atoms are never read. This makes the database self-contained, needs no
   rebuild of existing `.pt` files, and does not depend on the original
   structures still sitting at the paths recorded in each entry's `id`.

3. **The ranking key is the mean of the two length-normalised TM-scores**,
   `(tm_query + tm_target) / 2`, matching the paper. Both components are also
   reported so a user can re-sort by either.

4. **Exposed as `rfoldseek search --rerank tm`**, with `sw` the default so
   existing behaviour is untouched. TM mode does not run Smith-Waterman.

## Non-goals

- No E-value in TM mode. Karlin-Altschul parameters are calibrated against SW
  scores; they say nothing about TM-scores. TM mode prints no `bits`/`E-value`
  column rather than printing a meaningless one.
- No caching of TM-scores across runs.
- No vendoring of US-align source.
- No complex/multi-chain alignment; single chain per entry, as the database
  already assumes.
- No change to database format, and no rebuild of existing databases.

## Components

### `rfoldseek/usalign.py` (new)

The only module that knows US-align exists. Three public functions.

**`find_usalign(explicit=None) -> str`**

Resolution order: the `explicit` argument, then `$USALIGN`, then
`shutil.which("USalign")`. Raises `FileNotFoundError` with an actionable message
naming all three mechanisms and pointing at https://github.com/pylelab/USalign
when none resolves.

**`write_c3_pdb(coords, mask, path, chain="A") -> int`**

Writes one `ATOM` record per residue where `mask` is true, taking the C3'
channel (`coords[:, 0, :]`). Returns the number of residues written.

Fixed-width PDB record layout:

```
ATOM  %5d  C3' %3s %1s%4d    %8.3f%8.3f%8.3f  1.00  0.00           C
```

Atom name occupies columns 13-16 as `" C3'"`; residue name is right-justified
in columns 18-20. The database stores no sequence — `encode_entry` drops the
`seq` field — so the residue name is the placeholder `"  G"` for every residue
and molecule type is forced with `-mol RNA`. TM-score is computed from
coordinates alone, so the placeholder does not affect the result. Residues are
numbered 1..N over the *valid* residues, which is also the indexing the token
string uses, so a residue index in the output maps directly onto a token index.
The file ends with `TER` and `END` records.

**`score_pool(binary, query_coords, query_mask, candidates, min_residues=3) -> dict`**

`candidates` is a sequence of `(id, coords, mask)`. Returns
`{id: (tm_query, tm_target)}`.

1. Create a temporary directory (`tempfile.TemporaryDirectory`, removed on exit
   including on failure).
2. Write `query.pdb`, and one `<n>.pdb` per candidate under `targets/`, plus a
   list file naming them. Candidates with fewer than `min_residues` valid
   residues are skipped and counted; the caller reports the count.
   Candidate files are named by index, not by chain id, so ids containing
   spaces or slashes cannot break the list file; a local index -> id map
   translates the results back.
3. Run one subprocess for the whole pool:
   `USalign query.pdb -dir2 <tmp>/targets/ <tmp>/list.txt -mol RNA -outfmt 2`
4. Parse the tab-separated output. `-outfmt 2` emits a header line starting `#`
   followed by one row per pair with columns
   `PDBchain1 PDBchain2 TM1 TM2 RMSD ID1 ID2 IDali L1 L2 Lali`;
   `TM1` is normalised by chain 1 (the query), `TM2` by chain 2 (the target).
   Rows are matched back to candidates by the chain-2 filename.
5. Candidates with no row in the output get `(0.0, 0.0)` and a warning naming
   them.

The exact flag spellings above must be checked against the installed binary as
the first implementation step; they are written from the documented interface,
and no `USalign` is installed on the development machine to confirm them. If
`-dir2` turns out to need a different argument order, only this function
changes.

A non-zero exit status raises `RuntimeError` carrying the binary's stderr.

### `rfoldseek/search.py` (modified)

`Searcher.search` gains two keyword arguments: `rerank="sw"` and
`usalign=None`. The prefilter call and everything before it are unchanged.
After candidates are obtained:

- `rerank="sw"` — the existing path, byte-for-byte unchanged.
- `rerank="tm"` — resolve the binary via `find_usalign(usalign)`, call
  `score_pool` with the query and the candidates' stored `coords`/`mask` (read
  from `self.ents`, which `load_database` already returns), sort by
  `(tm_query + tm_target) / 2` descending, and emit hit dicts of the form
  `{"rank", "target", "tm_mean", "tm_query", "tm_target", "length"}`.
  No `sw_score`, no `bits`, no `evalue`.

Any other value of `rerank` raises `ValueError`.

`search_id` is unchanged; it already forwards keyword arguments, so
`s.search_id("chainA", rerank="tm")` works.

### `rfoldseek/cli.py` (modified)

Two new arguments on the `search` subparser:

| Option      | Description                                                       |
| ----------- | ----------------------------------------------------------------- |
| `--rerank`  | `sw` (default) or `tm`: rerank the candidate pool by token Smith-Waterman or by US-align TM-score |
| `--usalign` | Path to the `USalign` binary (default: `$USALIGN`, then `PATH`)   |

Both are forwarded to `search`/`search_id`. Output formatting branches on
whether `tm_mean` is present in the first hit:

```
# query: chainA  (128 tokens)   database: my_db.pt  (885 chains)   rerank: US-align TM
rank  target                  TM_mean  TM_query  TM_target    len
   1  4V9F_A                   0.7421    0.7550     0.7292    129
   2  6XU8_B                   0.5188    0.5044     0.5332    121
```

The SW table is unchanged. The `# query:` comment line gains a
`rerank: US-align TM` suffix in TM mode only, so existing SW output stays
byte-identical.

## Error handling

| Condition | Behaviour |
| --------- | --------- |
| `USalign` not found | `FileNotFoundError` naming `--usalign`, `$USALIGN`, `PATH`, and the project URL. From the CLI this surfaces as a one-line `SystemExit`, not a traceback. |
| US-align exits non-zero | `RuntimeError` carrying stderr |
| Candidate with < 3 valid residues | Skipped; count reported on a `#` comment line |
| Query with < 3 valid residues | `ValueError` before any subprocess is started |
| Pair missing from US-align output | TM recorded as 0.0, candidate named in a warning |
| No candidates survive the prefilter | Existing behaviour: the "no candidates" comment line, before any US-align work |

## Testing

Three unit tests need no `USalign` binary and run in CI:

1. `write_c3_pdb` round-trip: write a known `(L, 5, 3)` array, read it back with
   biotite, assert the coordinates equal the C3' channel of the valid residues
   and that masked-out residues are absent.
2. `-outfmt 2` parsing: feed a fixture string of real US-align tabular output to
   the parser and assert the id -> (TM1, TM2) mapping, including a row for a
   candidate that is absent.
3. Ranking: with `score_pool` monkeypatched to return fixed scores, assert
   `search(rerank="tm")` orders hits by the mean, populates all three TM fields,
   and emits no `evalue` key.

One integration test, skipped via `pytest.mark.skipif(shutil.which("USalign") is
None)`: build a tiny synthetic database with real coordinates and search it in TM
mode via `search` (not `search_id`, so the self-hit is retained), asserting
that every score lies in [0, 1] and that the entry aligned against itself
scores ~1.0.

Note for whoever runs these: the repository's existing tests do not run on
Python 3.9 (see Open questions), so a 3.10+ interpreter is required.

## Cost

`--top_n` defaults to 200, a value selected on the validation split for SW
reranking. US-align is orders of magnitude more expensive per candidate than
token Smith-Waterman, so TM mode does 200 structural superpositions per query.
The default is kept at 200 for consistency with the paper, and the README will
state the cost and suggest lowering `--top_n` for interactive use. Batching the
whole pool into a single subprocess removes per-candidate process overhead but
not the alignment cost itself.

## Open questions

1. **Should `--top_n` default lower in TM mode?** Proposed: no — keep 200 and
   document the cost. Revisit if measured latency is unacceptable.
2. **The package is broken on Python 3.9.** `pyproject.toml` declares
   `requires-python = ">=3.9"`, but `database.py:13` uses `str | None`, which is
   a syntax-level failure on 3.9 at import time. This is pre-existing and out of
   scope for RFoldseek-TM; it is recorded here because it blocks running the
   test suite on the current development machine. Fix separately by either
   raising the floor to 3.10 or adding `from __future__ import annotations`.

## Files touched

```
rfoldseek/usalign.py      new    binary discovery, PDB writing, batched scoring
rfoldseek/search.py       edit   rerank branch in Searcher.search
rfoldseek/cli.py          edit   --rerank / --usalign, TM output table
tests/test_usalign.py     new    unit tests 1-3 plus the skipped integration test
README.md                 edit   RFoldseek-TM section, new options, cost note
```
