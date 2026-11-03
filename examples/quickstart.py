"""End-to-end demo: human-in-PIG xenograft (Visium v1) -> cross-species dotplots.

Shows the full modern pipeline on a non-rat host:
  1. build a human<->pig cross-species CellPhoneDB database
  2. simulate a Visium v1 section (human graft in pig host) *before* deconvolution
  3. deconvolve each spot to per-cell-type pseudo-cells (abundance x signature)
  4. run the real CellPhoneDB via analyze_spatial (spatial microenvironments)
  5. read off pig<->human cross-species hits and draw CellPhoneDB-style dotplots
  6. write ready-to-fold multimer FASTAs for the structural (AlphaFold) filter

Run from the repo root:  python examples/quickstart.py
Outputs land in  examples/out/.

Swap `HOST` to "rat", "mouse", "rhesus", ... for a different host species.

NB the whole body is under ``if __name__ == "__main__"``: CellPhoneDB uses
multiprocessing (spawn on macOS), and worker processes re-import this module --
without the guard they would recursively re-run the pipeline and hang.
"""
import os

from species_cpdb.database import build_cross_species_db
from species_cpdb.synthetic import make_synthetic_visium_v1
from species_cpdb.preprocess import deconvolve_to_pseudocells
from species_cpdb.run import analyze_spatial
from species_cpdb.sequences import load_example_sequences
from species_cpdb.structure import write_fold_inputs
from species_cpdb import plotting

HOST = "pig"   # any registered species: rat, mouse, pig, rhesus, cynomolgus, ...


def main():
    out = os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)

    # 1. Cross-species database for human + pig (orthology confidence; try
    #    confidence="interface" for binding-residue-weighted scores).
    print(f"[1/6] building human<->{HOST} cross-species database ...")
    db = build_cross_species_db(
        "data/cpdb", f"data/xdb_{HOST}", species=["human", HOST], confidence="orthology"
    )
    print("      ", {k: db[k] for k in ("species", "interactions", "cross_species_interactions")})

    # 2. Simulate a Visium v1 human-graft-in-pig section (spots are mixtures).
    print("[2/6] simulating Visium v1 xenograft (pre-deconvolution) ...")
    spot, abundance, signature, celltype_species = make_synthetic_visium_v1(seed=0, host=HOST)

    # 3. Deconvolve spots -> (spot x cell type) pseudo-cells using the cell2location
    #    abundances + reference signature. Carries the region annotation along.
    print("[3/6] deconvolving spots to pseudo-cells ...")
    units = deconvolve_to_pseudocells(
        spot, abundance, signature, celltype_species, min_abundance=0.1, carry_obs=["region"]
    )
    print(f"       {spot.n_obs} spots -> {units.n_obs} pseudo-cells over {units.n_vars} genes")

    # 4. Run CellPhoneDB with spatial microenvironments (only co-located cell
    #    types tested). analyze_spatial composes scope + microenvs + the run.
    print("[4/6] running CellPhoneDB (spatial microenvironments) ...")
    res = analyze_spatial(
        units, db_zip=db["db_zip"], out_dir=os.path.join(out, "cpdb_pig"),
        scope="global", radius=2.0, n_niches=6, iterations=1000, threads=4, seed=1,
    )

    # 5. Cross-species hits + dotplots.
    hits = res.cross_species_hits(pvalue=0.05)
    print(f"\nCross-species (human<->{HOST}) significant interactions: {len(hits)}")
    print(hits[["interacting_pair", "cell_pair", "direction", "mean", "confidence"]].head(12).to_string(index=False))
    hits.to_csv(os.path.join(out, f"cross_species_hits_{HOST}.csv"), index=False)

    # ligand-class-aware variant: secreted ligands at diffusion range, juxtacrine
    # only at contact (uses CellPhoneDB's `secreted` flag).
    hits_lc = res.cross_species_hits(pvalue=0.05, ligand_class_proximity=True,
                                     radius_secreted=2.0, radius_contact=1.0)
    print(f"\nWith ligand-class-aware proximity: {len(hits_lc)} hits "
          f"({len(hits) - len(hits_lc)} juxtacrine pairs pruned to contact range)")

    plotting.dotplot(res, direction=f"{HOST}->human",
                     title=f"{HOST.capitalize()} ligand -> Human receptor (into graft)",
                     save=os.path.join(out, f"dotplot_{HOST}_to_human.png"))
    plotting.dotplot(res, direction=f"human->{HOST}",
                     title=f"Human ligand -> {HOST.capitalize()} receptor (out of graft)",
                     save=os.path.join(out, f"dotplot_human_to_{HOST}.png"))
    print(f"\nWrote hits + dotplots to {out}/")

    # 6. Structural filter inputs. The demo uses bundled example sequences so it
    #    runs offline; on your data leave `sequences` out and each hit's genes are
    #    fetched from UniProt for the right species (cached in ~/.cache/species_cpdb).
    #    To fold for real on a GPU:  run_structural_filter(hits, backend=ColabFoldBackend())
    print("[6/6] writing multimer FASTAs for the structural filter ...")
    if HOST == "pig":
        manifest = write_fold_inputs(hits, sequences=load_example_sequences(),
                                     out_dir=os.path.join(out, "fold_inputs"))
    else:
        manifest = write_fold_inputs(hits, out_dir=os.path.join(out, "fold_inputs"))  # UniProt
    ready = manifest["fasta"].notna().sum()
    print(f"       {ready}/{len(manifest)} cross-species complexes ready to fold -> {out}/fold_inputs/")


if __name__ == "__main__":
    main()
