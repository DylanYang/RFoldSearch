"""Alignment detail for the result page: traceback, identity, superposition.

The search itself only needs the best local Smith-Waterman score, so the
package computes it in linear memory without a traceback. The result page
needs the alignment itself -- which query and target positions pair up --
so here the same recurrence is run with full traceback matrices, for the
handful of hits a user is shown. Scoring is identical to
``rfoldseek.smith_waterman`` (same matrix, gap open -2.0, gap extend -0.5,
open cost charged on the first gapped position), and ``score`` is returned
so a caller can check it against the hit's ``sw_score``.

Positions are token indices, i.e. indices into the residues whose C3', C4'
and O4' are all present (the entry mask); ``masked_coords`` maps them onto
coordinates the same way.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:                                # pragma: no cover
    njit = None

GAP_OPEN = -2.0
GAP_EXTEND = -0.5
C4P = 1                    # coords[:, 1] is C4' (order: C3', C4', O4', C1', base)

# traceback codes
_STOP, _DIAG, _FROM_E, _FROM_F = 0, 1, 2, 3


def _sw_trace(q, t, sub, go, ge):
    lq, lt = len(q), len(t)
    # Scores need only the previous row; the traceback needs every cell, but
    # as int8 flags, so a 4,400 x 4,400 rRNA pair costs ~58 MB, not ~290 MB.
    Hprev = np.zeros(lt + 1, np.float32)
    Hcur = np.zeros(lt + 1, np.float32)
    F = np.full(lt + 1, -np.inf, np.float32)
    Hp = np.zeros((lq + 1, lt + 1), np.int8)       # how H[i, j] was reached
    Ee = np.zeros((lq + 1, lt + 1), np.int8)       # 1 = E extended, 0 = opened
    Fe = np.zeros((lq + 1, lt + 1), np.int8)
    best, bi, bj = 0.0, 0, 0
    for i in range(1, lq + 1):
        qi = q[i - 1]
        E = -np.inf
        Hcur[0] = 0.0
        for j in range(1, lt + 1):
            f_open = Hprev[j] + go
            f_ext = F[j] + ge
            if f_ext > f_open:
                F[j] = f_ext
                Fe[i, j] = 1
            else:
                F[j] = f_open
            e_open = Hcur[j - 1] + go
            e_ext = E + ge
            if e_ext > e_open:
                E = e_ext
                Ee[i, j] = 1
            else:
                E = e_open
            v = Hprev[j - 1] + sub[qi, t[j - 1]]
            p = _DIAG
            if E > v:
                v = E
                p = _FROM_E
            if F[j] > v:
                v = F[j]
                p = _FROM_F
            if v <= 0.0:
                v = 0.0
                p = _STOP
            Hcur[j] = v
            Hp[i, j] = p
            if v > best:
                best, bi, bj = v, i, j
        for j in range(lt + 1):
            Hprev[j] = Hcur[j]
    # walk back from the best cell; state 0 = H, 1 = E (gap in query), 2 = F
    qi_out = np.empty(lq + lt, np.int64)
    tj_out = np.empty(lq + lt, np.int64)
    n = 0
    i, j, state = bi, bj, 0
    while i > 0 and j > 0:
        if state == 0:
            p = Hp[i, j]
            if p == _STOP:
                break
            if p == _DIAG:
                qi_out[n] = i - 1
                tj_out[n] = j - 1
                n += 1
                i -= 1
                j -= 1
            elif p == _FROM_E:
                state = 1
            else:
                state = 2
        elif state == 1:                           # target residue vs gap
            qi_out[n] = -1
            tj_out[n] = j - 1
            n += 1
            if Ee[i, j] == 0:
                state = 0
            j -= 1
        else:                                      # query residue vs gap
            qi_out[n] = i - 1
            tj_out[n] = -1
            n += 1
            if Fe[i, j] == 0:
                state = 0
            i -= 1
    return best, qi_out[:n][::-1].copy(), tj_out[:n][::-1].copy()


if njit is not None:
    _sw_trace = njit(cache=True)(_sw_trace)


def sw_align(q, t, sub, go=GAP_OPEN, ge=GAP_EXTEND):
    """Best local alignment: (score, query index per column, target index per
    column), with -1 marking a gap."""
    return _sw_trace(np.asarray(q, np.int64), np.asarray(t, np.int64),
                     np.asarray(sub, np.float32), float(go), float(ge))


def masked_coords(entry):
    """(n_tokens, 5, 3) coordinates, one row per token."""
    coords = np.asarray(entry["coords"], np.float64)
    mask = np.asarray(entry["mask"]).astype(bool)
    return coords[mask]


def kabsch(P, Q):
    """Rotation R and translation so that P @ R.T + t best fits Q."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R, qc - pc @ R.T


def tm_d0(n):
    """RNA TM-score d0 (US-align's nucleic-acid form), in angstroms."""
    return max(0.5, 0.6 * np.sqrt(max(n, 1) - 0.5) - 2.5)


def describe(qtok, ttok, q_entry, t_entry, sub, q_seq=None, t_seq=None):
    """Everything the result page shows for one hit's alignment."""
    score, qi, tj = sw_align(qtok, ttok, sub)
    both = (qi >= 0) & (tj >= 0)
    pq, pt = qi[both], tj[both]
    n_pairs = int(both.sum())
    out = {
        "score": float(score),
        "n_aligned": n_pairs,
        "q_start": int(pq[0]) + 1 if n_pairs else 0,
        "q_end": int(pq[-1]) + 1 if n_pairs else 0,
        "t_start": int(pt[0]) + 1 if n_pairs else 0,
        "t_end": int(pt[-1]) + 1 if n_pairs else 0,
        "q_len": int(len(qtok)),
        "t_len": int(len(ttok)),
        "token_identity": float(np.mean(np.asarray(qtok)[pq] == np.asarray(ttok)[pt]))
        if n_pairs else 0.0,
    }
    # middle line of the 3Dr alignment, as in the paper's alignment figure:
    # '|' same token, '+' positive Structure-BLOSUM substitution, '.' other pair
    mid = []
    for a, b in zip(qi, tj):
        if a < 0 or b < 0:
            mid.append(" ")
        elif qtok[a] == ttok[b]:
            mid.append("|")
        elif sub[qtok[a], ttok[b]] > 0:
            mid.append("+")
        else:
            mid.append(".")
    out["tok_mid"] = "".join(mid)
    # A database without a sequence sidecar still gets alignment lines, with
    # the target's residues shown as placeholders and no sequence identity.
    if q_seq is not None:
        out["q_aln"] = "".join(q_seq[a] if a >= 0 else "-" for a in qi)
        out["t_aln"] = "".join((t_seq[b] if t_seq is not None else "\u00b7")
                               if b >= 0 else "-" for b in tj)
        out["t_seq_known"] = t_seq is not None
        if t_seq is not None:
            out["seq_identity"] = float(np.mean(
                [q_seq[a] == t_seq[b] for a, b in zip(pq, pt)])) if n_pairs else 0.0
    if n_pairs >= 3:
        Pq = masked_coords(q_entry)[pq, C4P]
        Pt = masked_coords(t_entry)[pt, C4P]
        R, tr = kabsch(Pt, Pq)
        d = np.linalg.norm(Pt @ R.T + tr - Pq, axis=1)
        out["rmsd"] = float(np.sqrt(np.mean(d ** 2)))
        # TM-like score of this superposition over the aligned pairs,
        # normalised by query length (the T_align of the Methods)
        out["tm_aln"] = float(np.sum(1.0 / (1.0 + (d / tm_d0(len(qtok))) ** 2))
                              / len(qtok))
        out["_superpose"] = (R, tr, pq, pt)
    return out


_ATOM_NAMES = ["C3'", "C4'", "O4'", "C1'", "N1"]
_LINK_MAX = 9.0            # C3'(i) - C4'(i+1) is ~6 A when the chain is continuous


def backbone_pdb(entry, seq, chain, rot=None, trans=None, highlight=None,
                 first_serial=1):
    """PDB records of the five stored atoms per token, optionally superposed.

    Only what the database stores can be drawn: C3', C4', O4', C1' and one
    base anchor per nucleotide, written as N1 so viewers bond it to C1'.
    Consecutive nucleotides are joined C3'(i) -> C4'(i+1) by CONECT records
    (the phosphate linkage itself is not stored); a gap longer than
    ``_LINK_MAX`` is left open. ``highlight`` (token indices) sets the
    B-factor to 1 for aligned residues and 0 elsewhere, for colouring.
    Returns (text, next free serial) so several chains can share a file.
    """
    xyz = masked_coords(entry)
    if rot is not None:
        xyz = xyz @ rot.T + trans
    marked = set(int(x) for x in highlight) if highlight is not None else set()
    lines, conect = [], []
    serial = first_serial
    prev_c3 = None                      # (serial, xyz) of the previous C3'
    for r, res in enumerate(xyz):
        name = (seq[r] if seq and r < len(seq) else "N").upper()
        name = name if name in "ACGU" else "N"
        b = 1.0 if r in marked else 0.0
        this_c3 = None
        for a, (x, y, z) in enumerate(res):
            if not np.all(np.isfinite((x, y, z))):
                continue
            atom = _ATOM_NAMES[a]
            lines.append(
                f"ATOM  {serial:5d} {atom:<4s} {name:>3s} {chain}{r + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           "
                f"{atom[0]:>2s}")
            if a == 1 and prev_c3 is not None and \
                    np.linalg.norm(prev_c3[1] - res[1]) < _LINK_MAX:
                conect.append(f"CONECT{prev_c3[0]:5d}{serial:5d}")
            if a == 0:
                this_c3 = (serial, res[0])
            serial += 1
        prev_c3 = this_c3
    lines.append("TER")
    return "\n".join(lines + conect) + "\n", serial
