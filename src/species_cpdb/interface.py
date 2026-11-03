"""Binding-interface conservation for the ``confidence="interface"`` model.

Whole-protein % identity (the ``orthology`` model) can mislead: what decides
whether a receptor still recognises a foreign ligand is conservation *at the
binding residues*, not across the whole chain. This module computes exactly that
and produces a **per-species** interface-conservation table that
:class:`species_cpdb.confidence.InterfaceConfidence` consumes.

Pipeline for real data (each step is network / data dependent, so provided as
pluggable providers):

1. **Interface residues** of a human protein -- the residues it uses to contact
   its partner -- from solved complex structures (PDBe/PISA, SIFTS mapping to
   UniProt numbering) or read off an AlphaFold model of the complex.
2. **Sequences** of the human protein and each species' ortholog (UniProt).
3. Align human vs ortholog and score the fraction of *interface* residues that
   are identical -> ``interface_pct_identity`` for that (gene, species).

The alignment + scoring (steps' math) is pure and unit-tested here; the fetchers
are thin, documented wrappers you point at your data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --- pure alignment + interface scoring (no external deps) ----------------

def _needleman_wunsch(a: str, b: str, match: int = 1, mismatch: int = -1, gap: int = -2):
    """Global alignment; returns two gap-padded strings. Fine for orthologs."""
    n, m = len(a), len(b)
    score = np.zeros((n + 1, m + 1))
    score[:, 0] = np.arange(n + 1) * gap
    score[0, :] = np.arange(m + 1) * gap
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            diag = score[i - 1, j - 1] + (match if ai == b[j - 1] else mismatch)
            score[i, j] = max(diag, score[i - 1, j] + gap, score[i, j - 1] + gap)
    # traceback
    ai, bj = [], []
    i, j = n, m
    while i > 0 and j > 0:
        s = score[i, j]
        if s == score[i - 1, j - 1] + (match if a[i - 1] == b[j - 1] else mismatch):
            ai.append(a[i - 1]); bj.append(b[j - 1]); i -= 1; j -= 1
        elif s == score[i - 1, j] + gap:
            ai.append(a[i - 1]); bj.append("-"); i -= 1
        else:
            ai.append("-"); bj.append(b[j - 1]); j -= 1
    while i > 0:
        ai.append(a[i - 1]); bj.append("-"); i -= 1
    while j > 0:
        ai.append("-"); bj.append(b[j - 1]); j -= 1
    return "".join(reversed(ai)), "".join(reversed(bj))


def interface_conservation(human_seq: str, ortholog_seq: str, interface_positions) -> tuple[float, int]:
    """% identity at the human interface residues, mapped onto the ortholog.

    ``interface_positions`` are 1-based residue positions in ``human_seq``. Returns
    ``(pct_identity_at_interface, n_interface_residues_mapped)``. If no interface
    residues map (e.g. all deleted in the ortholog), returns ``(nan, 0)``.
    """
    want = set(int(p) for p in interface_positions)
    ah, bo = _needleman_wunsch(human_seq, ortholog_seq)
    hpos = 0
    identical = mapped = 0
    for ch, co in zip(ah, bo):
        if ch != "-":
            hpos += 1
            if hpos in want and co != "-":
                mapped += 1
                identical += int(ch == co)
    if mapped == 0:
        return float("nan"), 0
    return 100.0 * identical / mapped, mapped


# --- build a per-species interface table ----------------------------------

def build_interface_table(
    genes: list[str],
    species: list[str],
    sequence_provider,
    interface_provider,
) -> pd.DataFrame:
    """Compute per-(gene, species) interface conservation.

    Parameters
    ----------
    sequence_provider
        ``callable(human_symbol, species) -> aa sequence`` ('human' for the
        reference). Wrap a UniProt fetch or a local FASTA lookup.
    interface_provider
        ``callable(human_symbol) -> list[int]`` of 1-based interface residue
        positions in the human sequence (from PDBe/PISA/SIFTS or an AF model).

    Returns
    -------
    DataFrame: ``species, human_symbol, interface_pct_identity, n_interface_residues``.
    """
    rows = []
    for g in genes:
        try:
            hseq = sequence_provider(g, "human")
            positions = interface_provider(g)
        except Exception:  # noqa: BLE001 -- skip genes without data
            continue
        if not hseq or not positions:
            continue
        for sp in species:
            if sp == "human":
                continue
            try:
                oseq = sequence_provider(g, sp)
            except Exception:  # noqa: BLE001
                continue
            if not oseq:
                continue
            pct, n = interface_conservation(hseq, oseq, positions)
            if n:
                rows.append({"species": sp, "human_symbol": g,
                             "interface_pct_identity": round(pct, 1), "n_interface_residues": n})
    return pd.DataFrame(rows, columns=["species", "human_symbol", "interface_pct_identity", "n_interface_residues"])


# --- thin, documented fetchers (network; optional) ------------------------

def fetch_uniprot_sequence(accession: str, timeout: int = 60) -> str:
    """Fetch a protein sequence from UniProt by accession (needs ``requests``)."""
    import requests

    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{accession}.fasta", timeout=timeout)
    r.raise_for_status()
    return "".join(r.text.splitlines()[1:])


def fetch_interface_residues_pdbe(pdb_id: str, uniprot: str, timeout: int = 60) -> list[int]:
    """Interface residues (UniProt numbering) for a chain in a PDB complex, via PDBe.

    Uses the PDBe 'interface residues' API and SIFTS to map to UniProt numbering.
    This is a template for the real fetch -- endpoints/derivation depend on the
    complex; adapt to your structures. Kept thin so it's easy to point at PISA,
    a local mmCIF, or interface residues read off an AlphaFold model instead.
    """
    raise NotImplementedError(
        "Point this at your interface-residue source (PDBe/PISA/SIFTS or an AF model). "
        "It must return 1-based interface positions in the human UniProt sequence; "
        "feed those (with sequences) to build_interface_table()."
    )
