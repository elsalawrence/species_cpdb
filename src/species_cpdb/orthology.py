"""Human -> species orthology via Ensembl BioMart (the production ortholog source).

The bundled seed (see :func:`species_cpdb.database.load_orthologs`) is only an
illustrative offline table for the demo. For real analyses, fetch the full map
with :func:`fetch_ensembl_orthologs`, which returns the **long format** the
database builder consumes:

    species, human_symbol, target_symbol, orthology_type, pct_identity

``pct_identity`` is the % identity of the *target* protein to its human ortholog
(``<stem>_homolog_perc_id``), read as "how likely is the native partner to still
recognise the foreign one". Any species in :mod:`species_cpdb.species` is
supported (mouse, rat, pig, rhesus, cynomolgus, marmoset, rabbit, dog, ...).

Requires network access and ``requests`` (``pip install .[orthology]``).
"""
from __future__ import annotations

import io

import pandas as pd

from .species import SPECIES, ensembl_homolog

# Ensembl BioMart martservice endpoint (accepts an XML query).
_BIOMART_URL = "https://www.ensembl.org/biomart/martservice"

_QUERY_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="hsapiens_gene_ensembl" interface="default">{filter}
    <Attribute name="external_gene_name"/>
    <Attribute name="{stem}_homolog_associated_gene_name"/>
    <Attribute name="{stem}_homolog_orthology_type"/>
    <Attribute name="{stem}_homolog_perc_id"/>
  </Dataset>
</Query>"""


def cpdb_interactome_genes(cpdb_dir: str) -> list[str]:
    """Ensembl gene ids of the CellPhoneDB interactome (to scope the ortholog fetch)."""
    import os

    gene = pd.read_csv(os.path.join(cpdb_dir, "gene_input.csv"))
    return sorted(set(gene["ensembl"].dropna().astype(str)))


def fetch_ensembl_orthologs(
    species: list[str] | str = ("mouse", "rat", "pig", "rhesus"),
    genes: list[str] | None = None,
    cache_path: str | None = None,
    batch_size: int = 500,
    timeout: int = 300,
) -> pd.DataFrame:
    """Fetch a **long-format** human->species ortholog table from Ensembl BioMart.

    Parameters
    ----------
    species
        Species name(s) from :mod:`species_cpdb.species` (not ``"human"``).
    genes
        Optional Ensembl gene ids to restrict to (e.g. from
        :func:`cpdb_interactome_genes` -- keeps the fetch to the ~1.8k genes the
        interactome uses). If None, fetches the whole genome (large, slow).
    cache_path
        If given, write the result to this CSV (and reuse it if it exists).

    Returns
    -------
    DataFrame: ``species, human_symbol, target_symbol, orthology_type, pct_identity, source``.
    """
    import os

    if cache_path and os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    if isinstance(species, str):
        species = [species]

    try:
        import requests
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "fetch_ensembl_orthologs needs `requests`; install with "
            "`pip install .[orthology]`, or use the bundled seed."
        ) from e

    batches = [None] if genes is None else [genes[i:i + batch_size] for i in range(0, len(genes), batch_size)]
    frames = []
    for sp in species:
        if sp not in SPECIES or sp == "human":
            raise ValueError(f"cannot fetch orthologs for {sp!r}")
        stem = ensembl_homolog(sp)
        for batch in batches:
            filt = "" if batch is None else f'\n    <Filter name="ensembl_gene_id" value="{",".join(batch)}"/>'
            query = _QUERY_TEMPLATE.format(stem=stem, filter=filt)
            resp = requests.post(_BIOMART_URL, data={"query": query}, timeout=timeout)
            resp.raise_for_status()
            raw = pd.read_csv(io.StringIO(resp.text), sep="\t")
            raw.columns = ["human_symbol", "target_symbol", "orthology_type", "pct_identity"]
            raw = raw.dropna(subset=["human_symbol", "target_symbol"])
            raw = raw[raw["target_symbol"].astype(str) != ""].copy()
            raw.insert(0, "species", sp)
            frames.append(raw)

    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    df["pct_identity"] = df["pct_identity"].round(1)
    df["source"] = "ensembl_biomart"
    if cache_path:
        df.to_csv(cache_path, index=False)
    return df


def merge_species_counts(human_adata, rat_adata, human_var_is_symbol: bool = True,
                         rat_var_is_symbol: bool = True):
    """Merge separate human and rat AnnData objects into one two-species object.

    Prefixes human genes with ``hsa:`` and rat genes with ``rno:`` and takes the
    outer union of features (a human cell has 0 for every ``rno:`` gene and vice
    versa). Both inputs must already share a normalised (unscaled) representation
    in ``.X`` and carry a common ``.obs`` schema (at least the label column you
    intend to pass to :func:`species_cpdb.run.run_analysis`).

    Use this to build the input for real data where human and rat reads were
    quantified against their own references.
    """
    import anndata as ad
    import numpy as np
    from ._namespace import add_prefix

    h = human_adata.copy()
    r = rat_adata.copy()
    h.var_names = [add_prefix(str(g), "human") for g in h.var_names]
    r.var_names = [add_prefix(str(g), "rat") for g in r.var_names]
    merged = ad.concat([h, r], join="outer", axis=0, fill_value=0.0)
    merged.obs = pd.concat([h.obs, r.obs], axis=0)
    return merged
