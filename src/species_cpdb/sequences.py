"""Protein sequences for the structural filter, fetched from UniProt and cached.

Folding a cross-species complex needs the amino-acid sequence of every chain,
in the right species (``rno:Tgfb1`` is the *rat* TGF-β1, ``hsa:TGFBR2`` the
human receptor). :func:`fetch_sequences` resolves each prefixed gene to a
UniProt entry by gene symbol + organism, downloads the sequence and keeps it in
a local JSON cache, so each gene is fetched once per machine.

Resolution, per gene:

1. reviewed (Swiss-Prot) entry whose gene name matches exactly;
2. else an unreviewed (TrEMBL) non-fragment entry -- flagged ``reviewed=False``.

The cache records the accession, so every sequence is traceable. The signal
peptide (cleaved before the protein reaches the cell surface) is stripped by
default; pro-domains (e.g. the TGF-β latency peptide) are not -- check those
by hand for complexes you care about.
"""
from __future__ import annotations

import json
import os
import warnings

from ._namespace import species_of, strip_prefix
from .species import taxon_id

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"


def default_cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "species_cpdb")


def _cache_path(cache_dir: str | None) -> str:
    return os.path.join(cache_dir or default_cache_dir(), "uniprot_sequences.json")


def load_cache(cache_dir: str | None = None) -> dict:
    fp = _cache_path(cache_dir)
    if os.path.exists(fp):
        with open(fp) as fh:
            return json.load(fh)
    return {}


def _save_cache(cache: dict, cache_dir: str | None) -> None:
    fp = _cache_path(cache_dir)
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    tmp = fp + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(cache, fh, indent=1, sort_keys=True)
    os.replace(tmp, fp)


def _signal_end(entry) -> int:
    """Last residue (1-based) of the annotated signal peptide, else 0."""
    for f in entry.get("features", []):
        if f.get("type") == "Signal":
            end = f.get("location", {}).get("end", {}).get("value")
            if isinstance(end, int):
                return end
    return 0


def _gene_names(entry) -> set:
    names = set()
    for g in entry.get("genes", []):
        if "geneName" in g:
            names.add(g["geneName"]["value"].upper())
    return names


def lookup_uniprot(symbol: str, species: str, timeout: int = 60) -> dict | None:
    """Best UniProt entry for a gene symbol in a species, or None.

    Returns ``{accession, sequence, signal_end, reviewed, gene}``.
    """
    import requests

    tax = taxon_id(species)
    for reviewed in (True, False):
        query = f"(gene_exact:{symbol}) AND (organism_id:{tax})"
        query += " AND (reviewed:true)" if reviewed else " AND (reviewed:false) AND (fragment:false)"
        r = requests.get(UNIPROT_SEARCH, timeout=timeout, params={
            "query": query, "format": "json", "size": 10,
            "fields": "accession,gene_names,sequence,ft_signal,reviewed",
        })
        r.raise_for_status()
        results = r.json().get("results", [])
        # gene_exact also matches synonyms; prefer an exact primary/any-name match
        exact = [e for e in results if symbol.upper() in _gene_names(e)] or results
        if exact:
            e = exact[0]
            return {
                "accession": e["primaryAccession"],
                "sequence": e["sequence"]["value"],
                "signal_end": _signal_end(e),
                "reviewed": reviewed,
                "gene": symbol,
            }
    return None


def fetch_sequences(
    genes,
    cache_dir: str | None = None,
    strip_signal_peptide: bool = True,
    refresh: bool = False,
    timeout: int = 60,
) -> dict:
    """``{prefixed_gene: amino-acid sequence}`` for prefixed ids like ``'rno:Tgfb1'``.

    Looks in the local cache first (``~/.cache/species_cpdb`` unless ``cache_dir``),
    queries UniProt for the rest and caches them. Genes that can't be resolved are
    left out with a warning (the structural filter then skips that complex).
    ``refresh=True`` re-queries everything.
    """
    cache = {} if refresh else load_cache(cache_dir)
    out, missing, fetched = {}, [], False
    for g in dict.fromkeys(genes):
        rec = cache.get(g)
        if rec is None:
            sp = species_of(g)
            if sp is None:
                missing.append(g)
                continue
            try:
                rec = lookup_uniprot(strip_prefix(g), sp, timeout=timeout)
            except Exception as e:  # noqa: BLE001 -- network errors shouldn't kill the run
                warnings.warn(f"UniProt lookup failed for {g}: {e}")
                missing.append(g)
                continue
            if rec is None:
                missing.append(g)
                continue
            cache[g] = rec
            fetched = True
        seq = rec["sequence"]
        if strip_signal_peptide and rec.get("signal_end"):
            seq = seq[rec["signal_end"]:]
        out[g] = seq
    if fetched:
        _save_cache(cache, cache_dir)
    if missing:
        warnings.warn(f"no UniProt sequence for {len(missing)} gene(s): {missing}")
    return out


def sequence_provenance(genes, cache_dir: str | None = None):
    """Accession / reviewed status of cached sequences, as a DataFrame."""
    import pandas as pd

    cache = load_cache(cache_dir)
    rows = [{"gene": g, **{k: v for k, v in cache[g].items() if k not in ("sequence", "gene")},
             "length": len(cache[g]["sequence"])} for g in genes if g in cache]
    return pd.DataFrame(rows)


def load_example_sequences() -> dict:
    """Bundled real UniProt sequences for the pig<->human quickstart hits (offline demo).

    Full precursors as fetched (signal peptides *not* stripped). For your own
    data use :func:`fetch_sequences`, which resolves whatever genes your hits contain.
    """
    from importlib import resources

    with resources.files("species_cpdb.data").joinpath("example_sequences.json").open() as fh:
        return json.load(fh)
