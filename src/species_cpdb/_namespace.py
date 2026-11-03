"""Unified two-species gene/protein namespace.

The whole trick that lets us reuse CellPhoneDB unchanged is to merge both
species into a *single* gene namespace. Every feature is prefixed by its
species so there are no symbol collisions:

    hsa:TGFB1   (human)
    rno:Tgfb1   (rat)

A merged counts matrix then holds the union of both prefixed genes. A human
spot expresses only ``hsa:`` genes (its ``rno:`` entries are 0) and vice
versa, which is exactly what CellPhoneDB's per-cell-type mean expects. A
cross-species interaction is then just an ordinary CellPhoneDB interaction
row whose partners happen to carry different prefixes, e.g.
``rno:Tgfb1 -> hsa:TGFBR2``.
"""
from __future__ import annotations

from . import species as _sp

SEP = ":"


class _PrefixView:
    """dict-like ``species -> prefix`` backed by the live species registry."""

    def __getitem__(self, name):
        return _sp.prefix_of(name)

    def __contains__(self, name):
        return name in _sp.SPECIES

    def __iter__(self):
        return iter(_sp.SPECIES)

    def keys(self):
        return _sp.SPECIES.keys()

    def items(self):
        return _sp.prefix_map().items()


# Backwards-compatible name: ``PREFIX[species] -> code`` for any registered species.
PREFIX = _PrefixView()


def add_prefix(symbol: str, species: str) -> str:
    """``add_prefix('TGFB1', 'human') -> 'hsa:TGFB1'``."""
    return f"{_sp.prefix_of(species)}{SEP}{symbol}"


def strip_prefix(identifier: str) -> str:
    """``strip_prefix('hsa:TGFB1') -> 'TGFB1'`` (unchanged if unprefixed)."""
    if SEP in identifier:
        return identifier.split(SEP, 1)[1]
    return identifier


def species_of(identifier: str) -> str | None:
    """Return the species name for a prefixed identifier, else None."""
    if SEP not in identifier:
        return None
    return _sp.species_of_prefix(identifier.split(SEP, 1)[0])
