"""Command-line interface for species_cpdb.

Subcommands:
  build-db      build the cross-species CellPhoneDB database from orthologs
  fetch-orthologs  download the human<->rat ortholog map from Ensembl
  run           run the cross-species spatial analysis on an AnnData
  demo          run the full synthetic end-to-end example
"""
from __future__ import annotations

import argparse
import os


def _cmd_fetch_orthologs(args):
    from .orthology import fetch_ensembl_orthologs, cpdb_interactome_genes
    from .species import SPECIES

    targets = ([s for s in SPECIES if s != "human"] if args.species == ["all"]
               else [s for s in args.species if s != "human"])
    genes = cpdb_interactome_genes(args.cpdb_dir) if args.cpdb_dir else None
    df = fetch_ensembl_orthologs(species=targets, genes=genes, cache_path=args.out)
    scope = f"{len(genes)} interactome genes" if genes else "whole genome"
    print(f"Fetched {len(df)} ortholog rows for {targets} ({scope}) -> {args.out}")


def _cmd_build_db(args):
    from .database import build_cross_species_db, load_orthologs

    orth = load_orthologs(args.orthologs) if args.orthologs else None
    summary = build_cross_species_db(
        args.cpdb_dir, args.out, orthologs=orth, species=args.species, confidence=args.confidence,
        orthologs_source=args.orthologs_source,
        restrict_to_orthologs=not args.keep_all, keep_human_only=args.keep_human_only,
    )
    print(summary)


def _cmd_run(args):
    import anndata as ad
    import pandas as pd
    from .run import analyze_spatial

    os.makedirs(args.out, exist_ok=True)
    if args.platform == "prepared":
        # counts is already an analysis-units AnnData (prefixed genes + obs['label'])
        units = ad.read_h5ad(args.counts)
    elif args.platform == "visium_hd" and not args.abundance:
        # bin2cell route: cell-level AnnData from b2c.bin_to_cell, cell types annotated
        from .preprocess import from_bin2cell

        if not args.celltype_key:
            raise SystemExit("--platform visium_hd (bin2cell) needs --celltype-key "
                             "(or pass --abundance/--signature for binned deconvolution)")
        species = args.species_key or args.hd_species
        if not species:
            raise SystemExit("--platform visium_hd needs --species-key (per-cell column) "
                             "or --hd-species (one species for the whole section)")
        cdata = ad.read_h5ad(args.counts)
        units = from_bin2cell(
            cdata, celltype_key=args.celltype_key, species=species,
            min_bins=args.min_bins, labels_source=args.labels_source, layer=args.layer,
            keep_genes=_db_genes(args.db),
            carry_obs=[args.region_key] if args.region_key in cdata.obs.columns else None)
        print(f"bin2cell: {units.n_obs} cells x {units.n_vars} database genes "
              f"(coordinates in {units.uns['spatial_units']})")
    elif args.platform in ("visium_v1", "visium_hd"):
        from .preprocess import deconvolve_to_pseudocells

        if not (args.abundance and args.signature):
            raise SystemExit(f"--platform {args.platform} needs --abundance and --signature CSVs")
        spot = ad.read_h5ad(args.counts)
        abundance = pd.read_csv(args.abundance, index_col=0)
        signature = pd.read_csv(args.signature, index_col=0)
        # cell-type -> species: from --celltype-species CSV, else inferred from the
        # 'species_celltype' key convention (e.g. human_fibroblast -> human).
        if args.celltype_species:
            cs = pd.read_csv(args.celltype_species)
            celltype_species = dict(zip(cs.iloc[:, 0], cs.iloc[:, 1]))
        else:
            celltype_species = {c: str(c).split("_", 1)[0] for c in signature.index}
        carry = [args.region_key] if args.region_key in spot.obs.columns else None
        units = deconvolve_to_pseudocells(spot, abundance, signature, celltype_species,
                                          min_abundance=args.min_abundance, carry_obs=carry)
    elif args.platform in ("xenium", "merfish", "cosmx"):
        from .preprocess import cells_as_units

        if not (args.celltype_key and args.species_key):
            raise SystemExit("--platform xenium needs --celltype-key and --species-key")
        adata = ad.read_h5ad(args.counts)
        units = cells_as_units(adata, celltype_key=args.celltype_key, species=args.species_key,
                               var_is_prefixed=args.var_is_prefixed)
    else:
        raise SystemExit(f"unknown --platform {args.platform!r}")
    adata = units
    if args.radius is None:
        # bin2cell coordinates are microns; spot/array grids are ~1 per spot
        args.radius = 30.0 if adata.uns.get("spatial_units") == "um" else 2.0
    res = analyze_spatial(
        adata, db_zip=args.db, out_dir=args.out, label_key=args.label,
        scope=args.scope, region=args.region, region_key=args.region_key,
        use_microenvs=not args.no_microenvs,
        engine=args.engine, radius=args.radius, n_niches=args.n_niches,
        iterations=args.iterations, threads=args.threads, pvalue=args.pvalue, seed=args.seed,
    )
    hits = res.cross_species_hits(pvalue=args.pvalue)
    hits_path = os.path.join(args.out, "cross_species_hits.csv")
    hits.to_csv(hits_path, index=False)
    print(f"{len(hits)} cross-species significant interactions ({args.engine} engine) -> {hits_path}")
    if args.plot:
        from . import plotting

        for d in sorted(set(hits["direction"])):
            try:
                plotting.dotplot(res, direction=d, save=os.path.join(args.out, f"dotplot_{d.replace('->', '_to_')}.png"))
            except ValueError:
                pass
        print(f"Dotplots -> {args.out}/")


def _db_genes(db_zip):
    """Prefixed gene names in the cross-species database (its gene_input.csv)."""
    import pandas as pd

    fp = os.path.join(os.path.dirname(db_zip), "gene_input.csv")
    return set(pd.read_csv(fp, usecols=["gene_name"])["gene_name"]) if os.path.exists(fp) else None


def _cmd_demo(args):
    import runpy

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    runpy.run_path(os.path.join(here, "examples", "quickstart.py"), run_name="__main__")


def main(argv=None):
    p = argparse.ArgumentParser(prog="species-cpdb", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    fo = sub.add_parser("fetch-orthologs", help="download human->species orthologs from Ensembl")
    fo.add_argument("--species", nargs="+", default=["all"],
                    help="target species, or 'all' (see species_cpdb.species; not 'human')")
    fo.add_argument("--cpdb-dir", help="scope the fetch to the CellPhoneDB interactome genes (recommended)")
    fo.add_argument("--out", default="data/orthologs_ensembl.csv")
    fo.set_defaults(func=_cmd_fetch_orthologs)

    bd = sub.add_parser("build-db", help="build the cross-species CellPhoneDB database")
    bd.add_argument("--cpdb-dir", required=True, help="dir with the released CellPhoneDB *_input.csv files")
    bd.add_argument("--out", required=True, help="output db directory")
    bd.add_argument("--species", nargs="+", default=["human", "rat"],
                    help="species to include, e.g. --species human mouse (human is always added)")
    bd.add_argument("--confidence", default="orthology", choices=["orthology", "interface"],
                    help="cross-species confidence model")
    bd.add_argument("--orthologs", help="ortholog CSV (overrides --orthologs-source)")
    bd.add_argument("--orthologs-source", default="ensembl", choices=["ensembl", "seed"],
                    help="bundled orthologs to use: 'ensembl' (real, all species) or 'seed' (demo)")
    bd.add_argument("--keep-all", action="store_true", help="do not restrict to ortholog-covered interactions")
    bd.add_argument("--keep-human-only", action="store_true", help="also keep human interactions lacking orthologs")
    bd.set_defaults(func=_cmd_build_db)

    rn = sub.add_parser("run", help="run cross-species spatial analysis")
    rn.add_argument("--counts", required=True,
                    help="AnnData .h5ad: prepared units (default), or the spot/cell matrix per --platform")
    rn.add_argument("--db", required=True, help="cross-species cellphonedb.zip")
    rn.add_argument("--out", required=True)
    rn.add_argument("--platform", default="prepared",
                    choices=["prepared", "visium_v1", "visium_hd", "xenium", "merfish", "cosmx"],
                    help="prepared = counts is already analysis units; else deconvolve/prepare here")
    # visium_v1/hd inputs (deconvolution done in-CLI)
    rn.add_argument("--abundance", help="[visium_v1] cell2location abundances CSV (spot x celltype)")
    rn.add_argument("--signature", help="[visium_v1] reference signature CSV (celltype x gene)")
    rn.add_argument("--celltype-species", help="[visium_v1] CSV (celltype,species); else inferred from prefix")
    rn.add_argument("--min-abundance", type=float, default=0.1, help="[visium_v1] min abundance per pseudo-cell")
    rn.add_argument("--region-key", default="region", help="obs column carrying region labels")
    # visium_hd via bin2cell (counts = cell-level AnnData from b2c.bin_to_cell)
    rn.add_argument("--hd-species", help="[visium_hd] one species for every cell (single-species section)")
    rn.add_argument("--min-bins", type=int, default=1, help="[visium_hd] drop cells with fewer 2um bins")
    rn.add_argument("--labels-source", nargs="+",
                    help="[visium_hd] keep cells whose <labels_key>_source is in these, e.g. primary")
    rn.add_argument("--layer", help="[visium_hd] use this layer instead of X (e.g. counts)")
    # single-cell platform inputs
    rn.add_argument("--celltype-key", help="[xenium/visium_hd] obs column with cell types")
    rn.add_argument("--species-key", help="[xenium/visium_hd] obs column with 'human'/'rat'/... per cell")
    rn.add_argument("--var-is-prefixed", action="store_true", help="[xenium] genes already hsa:/rno: prefixed")
    rn.add_argument("--label", default="label", help="obs column with species_celltype labels")
    rn.add_argument("--engine", default="celltype", choices=["celltype", "neighborhood"],
                    help="celltype = wrap CellPhoneDB (cell-type means); neighborhood = per-spot adjacency")
    rn.add_argument("--scope", default="global", choices=["global", "region", "window"])
    rn.add_argument("--region", nargs="+", help="region label(s) for --scope region (from obs['region'])")
    rn.add_argument("--radius", type=float, default=None,
                    help="neighbour radius in coordinate units (default 2 spots; 30 um for visium_hd bin2cell)")
    rn.add_argument("--n-niches", type=int, default=6)
    rn.add_argument("--no-microenvs", action="store_true", help="test all pairs (non-spatial)")
    rn.add_argument("--iterations", type=int, default=1000)
    rn.add_argument("--threads", type=int, default=4)
    rn.add_argument("--pvalue", type=float, default=0.05)
    rn.add_argument("--seed", type=int, default=0)
    rn.add_argument("--plot", action="store_true")
    rn.set_defaults(func=_cmd_run)

    dm = sub.add_parser("demo", help="run the synthetic end-to-end example")
    dm.set_defaults(func=_cmd_demo)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
