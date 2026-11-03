"""species_cpdb: cross-species spatial cell-cell interaction inference.

Wraps CellPhoneDB v5 to infer ligand-receptor interactions *between* species
(e.g. a rat ligand engaging a human receptor) in spatial transcriptomics data,
scored across a tissue border such as a xenograft margin.

Pipeline
--------
1. orthology  : build human <-> rat ortholog map (Ensembl, with a bundled seed).
2. database   : turn the CellPhoneDB human interactome + orthologs into a
                *cross-species* custom CellPhoneDB database.
3. spatial    : turn a spatial neighbour graph into CellPhoneDB microenvironments
                so only spatially co-located (cross-border) cell types are tested.
4. run        : call the real CellPhoneDB statistical method on the above.
5. plotting   : CellPhoneDB-style dotplots of the surviving cross-species pairs.
"""

from ._namespace import PREFIX, add_prefix, strip_prefix, species_of

# Lazy re-exports of the main API (kept lazy so `import species_cpdb` stays cheap
# and doesn't drag in CellPhoneDB/anndata unless you use them).
def __getattr__(name):
    _map = {
        "build_cross_species_db": "database",
        "load_orthologs": "database",
        "ortholog_map": "database",
        "fetch_ensembl_orthologs": "orthology",
        "get_confidence": "confidence",
        "build_interface_table": "interface",
        "interface_conservation": "interface",
        "deconvolve_to_pseudocells": "preprocess",
        "cells_as_units": "preprocess",
        "prepare_units": "preprocess",
        "from_bin2cell": "preprocess",
        "fetch_sequences": "sequences",
        "run_analysis": "run",
        "analyze_spatial": "run",
        "neighborhood_ligrec": "neighborhood",
        "filter_hits_structurally": "structure",
        "run_structural_filter": "structure",
        "write_fold_inputs": "structure",
        "ColabFoldBackend": "structure",
        "MockFoldingBackend": "structure",
    }
    if name in _map:
        import importlib
        return getattr(importlib.import_module(f".{_map[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .species import SPECIES, Species, register  # cheap, no heavy deps

__all__ = [
    "PREFIX", "add_prefix", "strip_prefix", "species_of",
    "SPECIES", "Species", "register",
    "build_cross_species_db", "load_orthologs", "ortholog_map", "fetch_ensembl_orthologs",
    "get_confidence",
    "deconvolve_to_pseudocells", "cells_as_units", "prepare_units", "from_bin2cell",
    "fetch_sequences",
    "run_analysis", "analyze_spatial", "neighborhood_ligrec",
    "filter_hits_structurally", "run_structural_filter", "write_fold_inputs",
    "ColabFoldBackend", "MockFoldingBackend",
]
__version__ = "0.1.0"
