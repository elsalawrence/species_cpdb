"""Pluggable cross-species confidence models.

Because there is no ground-truth cross-species interactome, every cross-species
L-R pair carries a confidence in ``0..1`` estimating whether the native partner
would still recognise the *foreign* (substituted) partner. Which evidence feeds
that estimate is a **choice**, exposed as ``confidence=`` on the database builder:

* ``"orthology"`` -- whole-protein % identity of the foreign partner (+ ortholog
  type as a fallback prior when identity is missing). Full coverage.
* ``"interface"`` -- conservation scored *only at the binding residues* (a
  conserved binding pocket is what matters for recognition, even if the rest of
  the protein diverged). Honest caveat: binding-residue annotations don't exist
  for most pairs, so it **falls back to orthology per gene** where absent, and
  records how many genes used interface data vs fell back.

New strategies (e.g. a structure-derived score) plug in by subclassing
:class:`ConfidenceModel` and registering in :data:`CONFIDENCE_MODELS`.
"""
from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd

# Fallback prior when a % protein identity is unavailable, by Ensembl ortholog
# type. Conservative: 1:1 ortholog is more trustworthy than 1:many, etc.
_ORTHOLOGY_TYPE_PRIOR = {
    "ortholog_one2one": 0.60,
    "ortholog_one2many": 0.40,
    "ortholog_many2many": 0.30,
}
_DEFAULT_PRIOR = 0.30


def identity_factor(pct_identity, orthology_type) -> float:
    """Per-gene conservation factor in ``0..1`` from % identity or a type prior."""
    if pct_identity is not None and not (isinstance(pct_identity, float) and np.isnan(pct_identity)):
        return float(pct_identity) / 100.0
    return _ORTHOLOGY_TYPE_PRIOR.get(orthology_type, _DEFAULT_PRIOR)


def load_interface_seed() -> pd.DataFrame:
    """Bundled *illustrative* interface-conservation table (see caveats in README)."""
    with resources.files("species_cpdb.data").joinpath("interface_seed.csv").open() as fh:
        return pd.read_csv(fh)


class ConfidenceModel:
    """Base class. ``partner_score`` scores one partner's genes in a species."""

    name = "base"

    def partner_score(self, human_genes: list[str], species: str, ortho_by_species: dict) -> tuple[float, str]:
        """Return ``(score in 0..1, provenance note)`` for a partner in ``species``.

        The reference species ``"human"`` scores 1.0 (native). Otherwise
        ``ortho_by_species[species]`` maps human symbol -> a record with
        ``.pct_identity`` and ``.orthology_type``.
        """
        raise NotImplementedError


class OrthologyConfidence(ConfidenceModel):
    name = "orthology"

    def partner_score(self, human_genes, species, ortho_by_species):
        if species == "human":
            return 1.0, "native"
        ortho = ortho_by_species.get(species, {})
        factors = [
            identity_factor(ortho[g].pct_identity, ortho[g].orthology_type)
            for g in human_genes if g in ortho
        ]
        if not factors:
            return _DEFAULT_PRIOR, "no-ortholog"
        return float(min(factors)), "orthology"


class InterfaceConfidence(ConfidenceModel):
    name = "interface"

    def __init__(self, interface_table: pd.DataFrame | None = None):
        if interface_table is None:
            interface_table = load_interface_seed()
        # Two accepted shapes:
        #  * per-species: columns [species, human_symbol, interface_pct_identity]
        #    (from interface.build_interface_table) -> looked up by (species, gene)
        #  * species-agnostic: [human_symbol, interface_pct_identity] (bundled seed)
        #    -> the same value applied to every species (illustrative)
        if "species" in interface_table.columns:
            self.per_species = True
            self.iface = {(r.species, r.human_symbol): r.interface_pct_identity
                          for r in interface_table.itertuples(index=False)}
        else:
            self.per_species = False
            self.iface = dict(zip(interface_table.human_symbol, interface_table.interface_pct_identity))

    def _lookup(self, gene, species):
        return self.iface.get((species, gene)) if self.per_species else self.iface.get(gene)

    def partner_score(self, human_genes, species, ortho_by_species):
        if species == "human":
            return 1.0, "native"
        ortho = ortho_by_species.get(species, {})
        factors, n_iface, n_fallback = [], 0, 0
        for g in human_genes:
            val = self._lookup(g, species)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                factors.append(float(val) / 100.0)
                n_iface += 1
            elif g in ortho:
                factors.append(identity_factor(ortho[g].pct_identity, ortho[g].orthology_type))
                n_fallback += 1
        if not factors:
            return _DEFAULT_PRIOR, "no-data"
        return float(min(factors)), f"interface({n_iface} iface, {n_fallback} fallback)"


CONFIDENCE_MODELS = {
    "orthology": OrthologyConfidence,
    "interface": InterfaceConfidence,
}


def get_confidence(name: str = "orthology", interface_table: pd.DataFrame | None = None) -> ConfidenceModel:
    """Factory: ``get_confidence('interface')`` etc."""
    if name not in CONFIDENCE_MODELS:
        raise ValueError(f"unknown confidence {name!r}; known: {list(CONFIDENCE_MODELS)}")
    if name == "interface":
        return InterfaceConfidence(interface_table)
    return CONFIDENCE_MODELS[name]()
