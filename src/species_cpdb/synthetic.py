"""Synthetic Visium-like spatial data for a human-in-rat cardiac xenograft.

Generates an AnnData that mimics what you'd have *after* cell2location: spots
on a grid, a species label per spot, a (dominant) cell-type label, spatial
coordinates, and a normalised expression matrix over a two-species gene
namespace (``hsa:`` / ``rno:``).

Geometry: a central human graft disk embedded in rat host tissue, with a border
band where the two tissues meet. Cross-border ligand-receptor signals are
planted deliberately so the downstream pipeline has something real to find:

    rat fibroblast   Tgfb1/Tgfb2  -> human  TGFBR1+TGFBR2   (fibrotic input to graft)
    human EC/CM      VEGFA        -> rat     Flt1/Kdr        (graft angiogenic signal)
    rat macrophage   Ccl2         -> human   CCR2            (host immune recruitment)
    human CM         NPPA/NPPB    -> rat     Npr1            (natriuretic signalling)

This is a *simulation for wiring/QC*, not biology. Swap in real data by
building an AnnData with the same fields (see the README).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad

from ._namespace import add_prefix
from .database import load_orthologs

# Role-based expression programs (genes are bare *human* symbols; the host
# species' orthologous symbols are substituted per its ortholog map). The graft
# is always human; the host can be any registered species (rat, pig, mouse, ...).
_GRAFT_PROGRAM = {  # human graft cells
    "cardiomyocyte": ["NPPA", "NPPB", "VEGFA", "TGFBR1", "TGFBR2", "IGF1R"],
    "endothelial": ["VEGFA", "KDR", "FLT1", "DLL4", "JAG1", "EFNB2", "ANGPT1"],
    "fibroblast": ["COL1A1", "PDGFRA", "PDGFRB", "TGFBR1", "TGFBR2", "IGF1"],
    "macrophage": ["CCR2", "TNF", "IL6", "TNFRSF1A"],
}
_HOST_PROGRAM = {  # host cells (secrete the ligands that engage the graft's receptors)
    "cardiomyocyte": ["NPR1", "TGFBR1", "TGFBR2", "IGF1R", "EPHB4"],
    "fibroblast": ["TGFB1", "TGFB2", "COL1A1", "PDGFRA", "PDGFRB", "IGF1"],
    "endothelial": ["FLT1", "KDR", "ANGPT1", "DLL4", "JAG1", "TEK"],
    "macrophage": ["CCL2", "TNF", "IL6", "IL1B", "TNFRSF1A", "CCR2"],
}


def _program(host: str) -> dict:
    """(species, cell_type) -> gene list, for human graft + the chosen host."""
    prog = {("human", ct): genes for ct, genes in _GRAFT_PROGRAM.items()}
    prog.update({(host, ct): genes for ct, genes in _HOST_PROGRAM.items()})
    return prog


def _composition(host: str) -> dict:
    return {
        "graft": {"human": {"cardiomyocyte": 0.5, "endothelial": 0.2, "fibroblast": 0.2, "macrophage": 0.1}},
        "host": {host: {"cardiomyocyte": 0.5, "fibroblast": 0.2, "endothelial": 0.2, "macrophage": 0.1}},
        "border": {
            "human": {"cardiomyocyte": 0.4, "endothelial": 0.3, "fibroblast": 0.2, "macrophage": 0.1},
            host: {"fibroblast": 0.4, "macrophage": 0.3, "endothelial": 0.2, "cardiomyocyte": 0.1},
        },
    }


def _all_symbols() -> list[str]:
    genes = set()
    for gl in list(_GRAFT_PROGRAM.values()) + list(_HOST_PROGRAM.values()):
        genes.update(gl)
    return sorted(genes)


def make_synthetic(
    grid: int = 40,
    graft_radius: float = 9.0,
    border_width: float = 2.5,
    seed: int = 0,
    high: float = 4.0,
    base: float = 0.15,
    host: str = "rat",
) -> ad.AnnData:
    """Build a synthetic human-graft-in-``host`` xenograft AnnData.

    ``host`` is any registered species (rat, pig, mouse, ...). Returns an AnnData
    with ``.obs['species','cell_type','region']``, ``.obsm['spatial']`` and a
    normalised (log1p CPM-like) two-species matrix.
    """
    rng = np.random.default_rng(seed)
    symbols = _all_symbols()
    program = _program(host)
    composition = _composition(host)
    # Map human symbol -> host ortholog symbol so host features are named exactly
    # as the cross-species database expects (e.g. ssc:VEGFA for pig).
    orth = load_orthologs(species=[host])
    h2t = dict(zip(orth.human_symbol, orth.target_symbol))

    def var_name(human_symbol: str, species: str) -> str:
        sym = human_symbol if species == "human" else h2t[human_symbol]
        return add_prefix(sym, species)

    # merged two-species feature space
    var_names = [var_name(s, "human") for s in symbols] + [var_name(s, host) for s in symbols]
    gene_idx = {g: i for i, g in enumerate(var_names)}

    # grid of spots
    xs, ys = np.meshgrid(np.arange(grid), np.arange(grid))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    centre = np.array([grid / 2, grid / 2])
    dist = np.linalg.norm(coords - centre, axis=1)

    region = np.where(
        dist <= graft_radius - border_width, "graft",
        np.where(dist <= graft_radius + border_width, "border", "host"),
    )

    n = coords.shape[0]
    species = np.empty(n, dtype=object)
    cell_type = np.empty(n, dtype=object)
    for i in range(n):
        reg = region[i]
        comp = composition[reg]
        sp = rng.choice(list(comp)) if len(comp) > 1 else next(iter(comp))
        species[i] = sp
        cts = comp[sp]
        cell_type[i] = rng.choice(list(cts), p=np.array(list(cts.values())))

    # expression: base everywhere for the species-appropriate genes, elevated
    # for the cell-type's program, Poisson-sampled, then normalised.
    X = np.zeros((n, len(var_names)), dtype=float)
    for i in range(n):
        sp = species[i]
        # a cell only expresses its own species' genes
        for s in symbols:
            X[i, gene_idx[var_name(s, sp)]] = base
        for s in program[(sp, cell_type[i])]:
            X[i, gene_idx[var_name(s, sp)]] = high
    X = rng.poisson(X).astype(float)

    adata = ad.AnnData(X=X)
    adata.var_names = var_names
    adata.obs_names = [f"spot_{i}" for i in range(n)]
    adata.obs["species"] = pd.Categorical(species)
    adata.obs["cell_type"] = pd.Categorical(cell_type)
    adata.obs["region"] = pd.Categorical(region)
    # CellPhoneDB label combines species + cell type
    adata.obs["label"] = pd.Categorical([f"{s}_{c}" for s, c in zip(species, cell_type)])
    adata.obsm["spatial"] = coords

    # library-size normalise + log1p (CellPhoneDB expects normalised, unscaled)
    libsize = adata.X.sum(1, keepdims=True)
    libsize[libsize == 0] = 1
    adata.layers["counts"] = adata.X.copy()
    adata.X = np.log1p(adata.X / libsize * 1e4)
    return adata


def _abund(host: str) -> dict:
    """Base cell abundances (cells/spot) per region, cell-type keys species-encoded."""
    return {
        "graft_core": {"human_cardiomyocyte": 3.0, "human_endothelial": 1.0,
                       "human_fibroblast": 1.0, "human_macrophage": 0.3},
        "graft_border": {"human_cardiomyocyte": 1.5, "human_endothelial": 1.0,
                         "human_fibroblast": 1.0, "human_macrophage": 0.3,
                         f"{host}_fibroblast": 1.5, f"{host}_macrophage": 1.0,
                         f"{host}_endothelial": 0.7, f"{host}_cardiomyocyte": 0.5},
        "host_distal": {f"{host}_cardiomyocyte": 3.0, f"{host}_fibroblast": 1.0,
                        f"{host}_endothelial": 1.0, f"{host}_macrophage": 0.5},
    }


def make_synthetic_visium_v1(
    grid: int = 34, graft_radius: float = 8.0, border_width: float = 2.5,
    seed: int = 0, high: float = 5.0, base: float = 0.2, total_scale: float = 1.0,
    dropout: float = 0.4, host: str = "rat",
):
    """Simulate a Visium v1 human-graft-in-``host`` xenograft *before* deconvolution.

    ``host`` is any registered species (rat, pig, mouse, ...). Returns
    ``(spot_adata, abundance, signature, celltype_species)`` where each spot's
    observed expression is a genuine mixture ``sum_c abundance*signature``
    (+ Poisson noise). Feed these to
    :func:`species_cpdb.preprocess.deconvolve_to_pseudocells`.
    """
    rng = np.random.default_rng(seed)
    orth = load_orthologs(species=[host])
    h2t = dict(zip(orth.human_symbol, orth.target_symbol))
    program = _program(host)
    abund = _abund(host)

    celltypes = sorted({ct for reg in abund.values() for ct in reg})
    celltype_species = {c: c.split("_", 1)[0] for c in celltypes}

    # build reference signature: cell-type x gene (species-appropriate symbols)
    def sym(gene, sp):
        return gene if sp == "human" else h2t[gene]

    genes = set()
    for c in celltypes:
        sp = celltype_species[c]
        ct = c.split("_", 1)[1]
        for g in program[(sp, ct)]:
            genes.add(sym(g, sp))
        for g in _all_symbols():
            genes.add(sym(g, sp))
    genes = sorted(genes)
    signature = pd.DataFrame(0.0, index=celltypes, columns=genes)
    for c in celltypes:
        sp = celltype_species[c]
        ct = c.split("_", 1)[1]
        for g in _all_symbols():
            signature.loc[c, sym(g, sp)] = base
        for g in program[(sp, ct)]:
            signature.loc[c, sym(g, sp)] = high

    # spot grid + regions
    xs, ys = np.meshgrid(np.arange(grid), np.arange(grid))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    centre = np.array([grid / 2, grid / 2])
    dist = np.linalg.norm(coords - centre, axis=1)
    region = np.where(dist <= graft_radius - border_width, "graft_core",
                      np.where(dist <= graft_radius + border_width, "graft_border", "host_distal"))
    n = coords.shape[0]

    # abundances per spot (base composition x lognormal noise). Each spot keeps
    # its dominant cell type but includes the rest only stochastically, so within
    # a spot only a *subset* of cell types co-occur (real spots aren't uniform);
    # this is what makes contact-range and diffusion-range proximity differ.
    abundance = pd.DataFrame(0.0, index=[f"spot_{i}" for i in range(n)], columns=celltypes)
    for i in range(n):
        for j, (c, a) in enumerate(abund[region[i]].items()):
            if j > 0 and rng.random() < dropout:
                continue
            abundance.iloc[i, abundance.columns.get_loc(c)] = a * total_scale * rng.lognormal(0, 0.4)

    # observed spot expression = mixture + Poisson noise, then normalised
    mix = abundance.values @ signature.values             # n_spots x genes
    counts = rng.poisson(mix).astype(float)
    lib = counts.sum(1, keepdims=True)
    lib[lib == 0] = 1
    Xnorm = np.log1p(counts / lib * 1e4)

    spot = ad.AnnData(X=Xnorm)
    spot.var_names = genes
    spot.obs_names = abundance.index
    spot.obs["region"] = pd.Categorical(region)
    spot.obsm["spatial"] = coords
    spot.layers["counts"] = counts
    return spot, abundance, signature, celltype_species


if __name__ == "__main__":
    a = make_synthetic()
    print(a)
    print(a.obs.groupby(["region", "species", "cell_type"], observed=True).size())
