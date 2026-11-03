"""CellPhoneDB-style dotplots for cross-species interactions.

Dot size encodes significance (-log10 p), colour encodes interaction magnitude
(mean of the two partners' expression), exactly as in the classic CellPhoneDB
dotplot. Because these are *predicted* cross-species pairs, the marker edge
additionally encodes our confidence in the pair (thicker/darker = higher).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def _long_table(result, pvalue: float, cross_only: bool) -> pd.DataFrame:
    # NeighborhoodResult (has a precomputed long table) or a raw long DataFrame
    if hasattr(result, "hits_long"):
        long = result.hits_long.copy()
        return long[long["cross_species"]].copy() if cross_only else long
    if isinstance(result, pd.DataFrame):
        return result[result["cross_species"]].copy() if cross_only else result.copy()
    # CpdbResult: build the long table from its wide means/pvalues
    pv = result.annotate()
    mn = result.means
    from .run import _META_COLS

    extra = ["species_a", "species_b", "cross_species", "direction", "confidence", "interactors"]
    id_cols = [c for c in pv.columns if c in _META_COLS or c in extra]
    pair_cols = [c for c in pv.columns if c not in id_cols]
    long = pv.melt(id_vars=id_cols, value_vars=pair_cols, var_name="cell_pair", value_name="pvalue")
    mlong = mn.melt(id_vars=[c for c in mn.columns if c in _META_COLS],
                    value_vars=pair_cols, var_name="cell_pair", value_name="mean")
    long = long.merge(mlong[["interacting_pair", "cell_pair", "mean"]],
                      on=["interacting_pair", "cell_pair"], how="left")
    if cross_only:
        long = long[long["cross_species"]].copy()
    return long


def dotplot(
    result,
    pvalue: float = 0.05,
    cross_only: bool = True,
    direction: str | None = None,
    cell_pairs: list[str] | None = None,
    interactions: list[str] | None = None,
    max_interactions: int = 25,
    max_cell_pairs: int = 25,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
    title: str | None = None,
    save: str | None = None,
):
    """Draw a CellPhoneDB-style dotplot of (cross-species) interactions.

    Parameters
    ----------
    direction
        Restrict to a single direction, e.g. ``'rat->human'`` or ``'human->rat'``.
    cell_pairs, interactions
        Optionally restrict/order the axes explicitly.
    max_interactions, max_cell_pairs
        Cap the axes to the strongest entries (by best -log10 p x confidence).
    """
    long = _long_table(result, pvalue, cross_only)
    if direction is not None:
        long = long[long["direction"] == direction].copy()
    if long.empty:
        raise ValueError("No interactions to plot after filtering.")

    long["neglog_p"] = -np.log10(long["pvalue"].clip(lower=1e-4))
    long["score"] = long["neglog_p"] * long["confidence"]

    # significant-in-at-least-one-pair interactions, ranked
    sig = long[long["pvalue"] <= pvalue]
    if interactions is None:
        order_i = (sig.groupby("interacting_pair")["score"].max()
                   .sort_values(ascending=False).index.tolist())
        interactions = order_i[:max_interactions]
    if cell_pairs is None:
        order_c = (sig[sig["interacting_pair"].isin(interactions)]
                   .groupby("cell_pair")["score"].max().sort_values(ascending=False).index.tolist())
        cell_pairs = order_c[:max_cell_pairs]

    sub = long[long["interacting_pair"].isin(interactions) & long["cell_pair"].isin(cell_pairs)].copy()
    sub = sub[sub["pvalue"] <= pvalue]  # only show significant dots
    if sub.empty:
        raise ValueError("No significant dots in the selected axes.")

    xcat = {c: i for i, c in enumerate(cell_pairs)}
    ycat = {p: i for i, p in enumerate(interactions)}
    sub["x"] = sub["cell_pair"].map(xcat)
    sub["y"] = sub["interacting_pair"].map(ycat)

    if figsize is None:
        figsize = (max(6, 0.55 * len(cell_pairs) + 4), max(4, 0.42 * len(interactions) + 1.5))
    fig, ax = plt.subplots(figsize=figsize)

    smin, smax = sub["neglog_p"].min(), sub["neglog_p"].max()
    sizes = 40 + 200 * (sub["neglog_p"] - smin) / (smax - smin + 1e-9)
    norm = Normalize(vmin=sub["mean"].min(), vmax=sub["mean"].max())
    sc = ax.scatter(
        sub["x"], sub["y"], s=sizes, c=sub["mean"], cmap=cmap, norm=norm,
        edgecolors=plt.get_cmap("Reds")(sub["confidence"].clip(0, 1).values),
        linewidths=0.5 + 2.5 * sub["confidence"].clip(0, 1).values,
    )

    ax.set_xticks(range(len(cell_pairs)))
    ax.set_xticklabels(cell_pairs, rotation=90, fontsize=8)
    ax.set_yticks(range(len(interactions)))
    ax.set_yticklabels(interactions, fontsize=8)
    ax.set_xlim(-0.5, len(cell_pairs) - 0.5)
    ax.set_ylim(-0.5, len(interactions) - 0.5)
    ax.margins(0.02)
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.set_title(title or ("Cross-species interactions" + (f" ({direction})" if direction else "")))

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label("mean expression\n(ligand, receptor)", fontsize=8)

    # A single combined key in the right margin (one block -> no cross-collision on
    # short plots): dot size = -log10(p), dot edge = cross-species confidence. Use
    # Line2D proxy handles (not ax.scatter) so no empty collections perturb the
    # tight-bbox extent calculation.
    from matplotlib.lines import Line2D

    def _ms(v):  # scatter area (points^2) -> marker diameter (points)
        return max(40.0, 40 + 200 * (v - smin) / (smax - smin + 1e-9)) ** 0.5

    handles, labels = [], []
    p_vals = sorted({round(v, 1) for v in np.linspace(smin, smax, 3)})
    for v in p_vals:  # collapses to one entry when every hit is maximally significant
        handles.append(Line2D([], [], marker="o", linestyle="none", markersize=_ms(v),
                              markerfacecolor="lightgrey", markeredgecolor="grey"))
        labels.append(f"{v:.1f}" + (" (max)" if len(p_vals) == 1 else ""))
    handles.append(Line2D([], [], linestyle="none")); labels.append("")  # spacer
    for c, lbl in ((0.4, "conf. low"), (0.95, "conf. high")):
        handles.append(Line2D([], [], marker="o", linestyle="none", markersize=12,
                              markerfacecolor="lightgrey",
                              markeredgecolor=plt.get_cmap("Reds")(c), markeredgewidth=0.5 + 2.5 * c))
        labels.append(lbl)
    leg = ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.16, 0.5),
                    title="size = -log10(p)\nedge = x-species conf.", frameon=False,
                    labelspacing=1.3, borderpad=0.6, handletextpad=0.9,
                    fontsize=8, title_fontsize=8)
    leg.get_title().set_ha("left")
    ax.add_artist(leg)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig, ax


def interaction_heatmap(result, direction: str | None = None, pvalue: float = 0.05, save: str | None = None):
    """Number of significant cross-species interactions per cell-type pair."""
    long = _long_table(result, pvalue, cross_only=True)
    if direction is not None:
        long = long[long["direction"] == direction]
    sig = long[long["pvalue"] <= pvalue]
    counts = sig.groupby("cell_pair").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(5, 0.4 * len(counts) + 2), 4))
    ax.bar(range(len(counts)), counts.values, color="steelblue")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=90, fontsize=7)
    ax.set_ylabel("# significant cross-species interactions")
    ax.set_title("Cross-border interaction load" + (f" ({direction})" if direction else ""))
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig, ax
