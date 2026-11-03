"""Registry of supported species.

The tool is species-agnostic: the human CellPhoneDB interactome is the
reference, and any species is defined by its **human ortholog map**. Each
species has a short KEGG-style ``prefix`` (used in the merged gene namespace,
e.g. ``hsa:``, ``mmu:``) and, for the real Ensembl fetch, the BioMart homolog
attribute stem (e.g. ``mmusculus`` -> ``mmusculus_homolog_perc_id``).

Add your own with :func:`register`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Species:
    name: str            # canonical name used throughout the API ('mouse')
    prefix: str          # KEGG-style 3-letter code, unique ('mmu')
    ensembl_homolog: str  # BioMart homolog attribute stem ('' for the human reference)
    description: str = ""
    taxon_id: int | None = None  # NCBI taxonomy id (UniProt sequence lookup)


# Common research / xenograft-host species. Reference = human.
_SPECIES = [
    Species("human", "hsa", "", "Homo sapiens (reference)", 9606),
    Species("mouse", "mmu", "mmusculus", "Mus musculus", 10090),
    Species("rat", "rno", "rnorvegicus", "Rattus norvegicus", 10116),
    Species("pig", "ssc", "sscrofa", "Sus scrofa", 9823),
    Species("rhesus", "mcc", "mmulatta", "Macaca mulatta (rhesus macaque)", 9544),
    Species("cynomolgus", "mcf", "mfascicularis", "Macaca fascicularis (cynomolgus macaque)", 9541),
    Species("marmoset", "cja", "cjacchus", "Callithrix jacchus (common marmoset)", 9483),
    Species("rabbit", "ocu", "ocuniculus", "Oryctolagus cuniculus", 9986),
    Species("dog", "cfa", "clfamiliaris", "Canis lupus familiaris", 9615),
    Species("sheep", "oas", "oaries", "Ovis aries", 9940),
    Species("cow", "bta", "btaurus", "Bos taurus", 9913),
    Species("zebrafish", "dre", "drerio", "Danio rerio", 7955),
]

SPECIES: dict[str, Species] = {s.name: s for s in _SPECIES}
_BY_PREFIX: dict[str, str] = {s.prefix: s.name for s in _SPECIES}


def register(species: Species) -> None:
    """Add or override a species in the registry."""
    SPECIES[species.name] = species
    _BY_PREFIX[species.prefix] = species.name


def prefix_of(name: str) -> str:
    if name not in SPECIES:
        raise KeyError(f"unknown species {name!r}; known: {sorted(SPECIES)}")
    return SPECIES[name].prefix


def species_of_prefix(prefix: str) -> str | None:
    return _BY_PREFIX.get(prefix)


def prefix_map() -> dict[str, str]:
    """name -> prefix for every registered species (kept live)."""
    return {name: s.prefix for name, s in SPECIES.items()}


def ensembl_homolog(name: str) -> str:
    return SPECIES[name].ensembl_homolog


def taxon_id(name: str) -> int:
    tid = SPECIES[name].taxon_id
    if tid is None:
        raise KeyError(f"species {name!r} has no taxon_id; register it with Species(..., taxon_id=...)")
    return tid
