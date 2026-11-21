"""The base-frame cache in edges.py must not change any edge output."""
from pathlib import Path

import numpy as np
import pytest

from rfoldsearch import edges

EXAMPLES = sorted((Path(__file__).resolve().parents[1]
                   / "webserver" / "static" / "examples").glob("*.cif"))
needs_cache = pytest.mark.skipif(edges._ORIG_MATCH_BASE is None,
                                 reason="biotite _match_base not available")


@pytest.mark.skipif(not EXAMPLES, reason="example structures not present")
@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_cached_edges_match_uncached(path, monkeypatch):
    monkeypatch.setattr(edges, "MEMOISE_BASE_FRAMES", False)
    plain = edges.find_base_pair_and_stack_idx(str(path))
    monkeypatch.setattr(edges, "MEMOISE_BASE_FRAMES", True)
    cached = edges.find_base_pair_and_stack_idx(str(path))
    assert plain.keys() == cached.keys()
    for key in plain:
        assert np.array_equal(plain[key], cached[key]), key


@needs_cache
@pytest.mark.skipif(not EXAMPLES, reason="example structures not present")
def test_cache_fits_each_nucleotide_once(monkeypatch):
    calls = []
    real = edges._ORIG_MATCH_BASE

    def counting(nucleotide, min_atoms_per_base):
        calls.append(1)
        return real(nucleotide, min_atoms_per_base)

    monkeypatch.setattr(edges, "_ORIG_MATCH_BASE", counting)
    path = str(next(p for p in EXAMPLES if p.stem == "1EHZ_A"))
    out = edges.find_base_pair_and_stack_idx(path)
    n_res = len(out["stack_idx"])
    # base_pairs and base_stacking each pass their own view of a nucleotide,
    # so it is fitted at most once per caller instead of once per candidate
    # pair (about 18 times per residue without the cache).
    assert 0 < len(calls) <= 2 * n_res


@needs_cache
def test_cache_hands_out_copies_and_is_scoped():
    import biotite.structure as struc

    atoms = [struc.Atom([float(i), 0.5 * i, 0.0], chain_id="A", res_id=1,
                        res_name="A", atom_name=name, element=name[0])
             for i, name in enumerate(
                 ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"])]
    nuc = struc.array(atoms)
    with edges._memoised_base_frames():
        first = edges._match_base_cached(nuc, 3)
        if first is None:
            pytest.skip("synthetic base not matched")
        expected = first.copy()
        first[:] = 0.0                              # caller mutates its copy
        second = edges._match_base_cached(nuc, 3)
        assert np.array_equal(second, expected)
    assert getattr(edges._local, "cache", None) is None
