"""Fast smoke tests for the cross-species spatial pipeline.

Run: pytest -q   (needs the CellPhoneDB reference db in data/cpdb; the demo
downloads it once, or see the README).
"""
import os

import pytest

CPDB_DIR = "data/cpdb"
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CPDB_DIR, "interaction_input.csv")),
    reason="reference CellPhoneDB db not present in data/cpdb",
)


def test_build_and_annotate(tmp_path):
    from species_cpdb.database import build_cross_species_db
    from species_cpdb.synthetic import make_synthetic
    from species_cpdb.spatial import build_microenvironments, cross_species_niches
    from species_cpdb.run import run_analysis

    db = build_cross_species_db(CPDB_DIR, str(tmp_path / "xdb"))
    assert db["cross_species_interactions"] > 0

    adata = make_synthetic(seed=0, grid=24)  # smaller grid for speed
    me = build_microenvironments(adata, radius=2.0, n_niches=5)
    assert cross_species_niches(me), "expected at least one mixed-species niche"

    res = run_analysis(
        adata, db_zip=db["db_zip"], out_dir=str(tmp_path / "run"),
        microenvs=me, iterations=100, threads=2, seed=1,
    )
    hits = res.cross_species_hits(pvalue=0.05)
    assert len(hits) > 0
    # the planted rat-fibroblast TGFB -> human receptor signal should surface
    assert hits["interacting_pair"].str.contains("Tgfb").any()
    assert set(hits["direction"]).issubset({"rat->human", "human->rat"})


def test_visium_v1_deconvolution(tmp_path):
    """v1 path: deconvolve spots -> pseudo-cells -> cross-species hits, + region scope."""
    from species_cpdb.database import build_cross_species_db
    from species_cpdb.synthetic import make_synthetic_visium_v1
    from species_cpdb.preprocess import deconvolve_to_pseudocells
    from species_cpdb.run import analyze_spatial

    db = build_cross_species_db(CPDB_DIR, str(tmp_path / "xdb"))
    spot, ab, sig, cts = make_synthetic_visium_v1(seed=0, grid=24)
    units = deconvolve_to_pseudocells(spot, ab, sig, cts, min_abundance=0.1, carry_obs=["region"])
    # pseudo-cells partition the spot: many more units than spots, genes prefixed
    assert units.n_obs > spot.n_obs
    assert all(":" in v for v in units.var_names)

    res = analyze_spatial(units, db_zip=db["db_zip"], out_dir=str(tmp_path / "run"),
                          radius=1.5, n_niches=5, iterations=100, threads=2, seed=1)
    hits = res.cross_species_hits(pvalue=0.05)
    assert hits["interacting_pair"].str.contains("Tgfb").any()

    # region scope restricts without error and still finds cross-species signal
    res_b = analyze_spatial(units, db_zip=db["db_zip"], out_dir=str(tmp_path / "run_b"),
                            scope="region", region="graft_border", region_key="region",
                            radius=1.5, n_niches=5, iterations=100, threads=2, seed=1)
    assert len(res_b.cross_species_hits(pvalue=0.05)) > 0


def test_neighborhood_engine(tmp_path):
    """Native per-spot adjacency engine finds the planted signal + directions."""
    from species_cpdb.database import build_cross_species_db
    from species_cpdb.synthetic import make_synthetic_visium_v1
    from species_cpdb.preprocess import deconvolve_to_pseudocells
    from species_cpdb.run import analyze_spatial

    db = build_cross_species_db(CPDB_DIR, str(tmp_path / "xdb"))
    spot, ab, sig, cts = make_synthetic_visium_v1(seed=0, grid=22)
    units = deconvolve_to_pseudocells(spot, ab, sig, cts, min_abundance=0.1)

    res = analyze_spatial(units, db_zip=db["db_zip"], out_dir=str(tmp_path / "run"),
                          engine="neighborhood", radius=1.5, iterations=100, seed=1)
    hits = res.cross_species_hits(pvalue=0.05)
    assert len(hits) > 0
    assert hits["interacting_pair"].str.contains("Tgfb").any()
    assert set(hits["direction"]).issubset({"rat->human", "human->rat"})
    assert (hits["n_edges"] > 0).all()


def test_confidence_models_and_fallback():
    """orthology vs interface scoring; interface falls back per-gene."""
    import collections
    from species_cpdb.confidence import get_confidence

    Rec = collections.namedtuple("Rec", "pct_identity orthology_type")
    obs = {"rat": {"TGFBR1": Rec(80, "ortholog_one2one"), "TGFBR2": Rec(78, "ortholog_one2one")}}

    o = get_confidence("orthology")
    i = get_confidence("interface")
    assert o.partner_score(["TGFBR1"], "human", obs) == (1.0, "native")   # reference is native
    so, no = o.partner_score(["TGFBR1", "TGFBR2"], "rat", obs)
    si, ni = i.partner_score(["TGFBR1", "TGFBR2"], "rat", obs)
    assert 0 <= so <= 1 and 0 <= si <= 1
    assert "interface" in ni and "0 fallback" in ni      # both genes in interface seed
    # a gene absent from the interface table falls back to orthology
    sf, nf = i.partner_score(["COL1A1"], "rat", {"rat": {"COL1A1": Rec(90, "ortholog_one2one")}})
    assert "fallback" in nf and abs(sf - 0.90) < 1e-9


def test_interface_conservation_and_per_species_confidence():
    """Interface residues score conservation at the binding site, not whole protein."""
    from species_cpdb.interface import interface_conservation, build_interface_table
    from species_cpdb.confidence import InterfaceConfidence
    import collections, pandas as pd

    human = "MAAAKKKAAA"          # interface residues at positions 5,6,7 = K,K,K
    good = "MAAAKKKAAA"           # identical interface
    bad = "MAAAEEEAAA"           # interface KKK -> EEE (0% conserved) but rest identical
    pct_good, n = interface_conservation(human, good, [5, 6, 7])
    pct_bad, _ = interface_conservation(human, bad, [5, 6, 7])
    assert n == 3 and pct_good == 100.0 and pct_bad == 0.0

    # per-species interface table feeds InterfaceConfidence keyed by (species, gene)
    tbl = pd.DataFrame({"species": ["pig", "mouse"], "human_symbol": ["KDR", "KDR"],
                        "interface_pct_identity": [95.0, 60.0]})
    m = InterfaceConfidence(tbl)
    Rec = collections.namedtuple("Rec", "pct_identity orthology_type")
    obs = {"pig": {"KDR": Rec(80, "ortholog_one2one")}, "mouse": {"KDR": Rec(80, "ortholog_one2one")}}
    sp_pig, _ = m.partner_score(["KDR"], "pig", obs)
    sp_mouse, _ = m.partner_score(["KDR"], "mouse", obs)
    assert abs(sp_pig - 0.95) < 1e-9 and abs(sp_mouse - 0.60) < 1e-9   # per-species differs


def test_ligand_class_proximity_discriminates():
    """Contact-range excludes near-but-untouching pairs; diffusion includes them."""
    import numpy as np, pandas as pd, anndata as ad
    from species_cpdb.spatial import celltype_proximity

    u = ad.AnnData(X=np.zeros((3, 1)))
    u.obs["label"] = pd.Categorical(["human_A", "rat_C", "rat_B"])
    u.obsm["spatial"] = np.array([[0, 0], [0, 0], [1.5, 0]], float)  # C touches A; B is 1.5 away

    contact = celltype_proximity(u, "label", radius=1.0)
    diffuse = celltype_proximity(u, "label", radius=2.0)
    AB = frozenset(("human_A", "rat_B"))
    assert AB not in contact and AB in diffuse


def test_structure_filter_stub():
    import numpy as np, pandas as pd
    import pytest as _pt
    from species_cpdb.structure import (filter_hits_structurally, example_scores_template,
                                         write_fold_inputs)

    hits = pd.DataFrame({"interacting_pair": ["hsa:A_ssc:B", "hsa:C_ssc:D", "hsa:E_ssc:F"],
                         "cross_species": [True, True, True], "mean": [1, 2, 3]})
    scores = example_scores_template(hits)
    scores["iptm"] = [0.9, 0.3, 0.7]                            # other metrics stay NaN
    # NaN metrics (pae/pdockq) are ignored; only iptm judged -> b dropped
    kept = filter_hits_structurally(hits, scores=scores, min_iptm=0.5)
    assert set(kept["interacting_pair"]) == {"hsa:A_ssc:B", "hsa:E_ssc:F"}
    # affinity cutoff on delta_g (more negative = tighter): keep <= -6
    scores["delta_g"] = [-8.0, -2.0, -7.0]
    kept2 = filter_hits_structurally(hits, scores=scores, min_iptm=None, max_delta_g=-6.0)
    assert set(kept2["interacting_pair"]) == {"hsa:A_ssc:B", "hsa:E_ssc:F"}
    with _pt.raises(NotImplementedError):
        filter_hits_structurally(hits)                          # no scores -> stub raises
    with _pt.raises(NotImplementedError):
        filter_hits_structurally(hits, predictor=lambda r: {})  # on-the-fly not wired


def test_run_structural_filter_mock(tmp_path):
    """The tool-runs-folding path: mock backend -> scores -> cutoff filter."""
    import pandas as pd
    from species_cpdb.structure import run_structural_filter, MockFoldingBackend

    hits = pd.DataFrame({
        "interacting_pair": ["ssc:TGFB1_hsa:TGFBR1+hsa:TGFBR2", "ssc:IL6_hsa:IL6R"],
        "cross_species": [True, True],
    })
    seqs = {"ssc:TGFB1": "MAAA", "hsa:TGFBR1": "MBBB", "hsa:TGFBR2": "MCCC",
            "ssc:IL6": "MDDD", "hsa:IL6R": "MEEE"}
    backend = MockFoldingBackend(scores_by_pair={
        "ssc:TGFB1_hsa:TGFBR1+hsa:TGFBR2": {"iptm": 0.85},   # confident -> kept
        "ssc:IL6_hsa:IL6R": {"iptm": 0.30},                  # low -> dropped
    })
    filtered, scores = run_structural_filter(
        hits, seqs, str(tmp_path / "sf"), backend=backend, device="cpu",
        run_prodigy=False, min_iptm=0.5, min_pdockq=None, max_interface_pae=None)
    assert set(scores["interacting_pair"]) == set(hits["interacting_pair"])
    assert set(filtered["interacting_pair"]) == {"ssc:TGFB1_hsa:TGFBR1+hsa:TGFBR2"}


def test_write_fold_inputs(tmp_path):
    import pandas as pd
    from species_cpdb.structure import write_fold_inputs

    hits = pd.DataFrame({"interacting_pair": ["ssc:TGFB1_hsa:TGFBR1+hsa:TGFBR2"],
                         "cross_species": [True]})
    seqs = {"ssc:TGFB1": "MABC", "hsa:TGFBR1": "MDEF", "hsa:TGFBR2": "MGHI"}
    man = write_fold_inputs(hits, seqs, str(tmp_path))
    assert man.loc[0, "n_chains"] == 3 and man.loc[0, "missing_sequences"] == ""
    text = open(man.loc[0, "fasta"]).read()
    assert text.count(">") == 3 and "MABC" in text  # 3-chain multimer FASTA written


@pytest.mark.skipif(
    not os.path.exists(os.path.join(CPDB_DIR, "interaction_input.csv")),
    reason="reference CellPhoneDB db not present",
)
def test_multi_species_db(tmp_path):
    """Any species (mouse/pig/...) and >2 species at once produce cross pairs."""
    import pandas as pd
    from species_cpdb.database import build_cross_species_db

    sm = build_cross_species_db(CPDB_DIR, str(tmp_path / "mouse"), species=["human", "mouse"])
    assert sm["cross_species_interactions"] > 0 and sm["species"] == ["human", "mouse"]
    meta = pd.read_csv(os.path.join(str(tmp_path / "mouse"), "interaction_meta.csv"))
    assert meta["partner_a_id"].str.startswith("mmu").any()   # mouse-prefixed partners exist

    # three species -> all 9 direction combos incl. a non-human/non-human cross
    s3 = build_cross_species_db(CPDB_DIR, str(tmp_path / "three"), species=["human", "pig", "rhesus"])
    m3 = pd.read_csv(os.path.join(str(tmp_path / "three"), "interaction_meta.csv"))
    dirs = set(m3["species_a"] + "->" + m3["species_b"])
    assert "pig->rhesus" in dirs and "rhesus->pig" in dirs


def test_real_ensembl_orthologs_bundled():
    """The bundled real Ensembl ortholog table covers many species and genes."""
    from species_cpdb.database import load_orthologs

    allo = load_orthologs(source="ensembl")
    assert len(allo) > 5000                        # whole-interactome, many species
    assert {"pig", "mouse", "rhesus", "zebrafish"} <= set(allo["species"])
    pig = load_orthologs(source="ensembl", species=["pig"])
    assert (pig["species"] == "pig").all() and "TGFB1" in set(pig["human_symbol"])


@pytest.mark.skipif(
    not os.path.exists(os.path.join(CPDB_DIR, "interaction_input.csv")),
    reason="reference CellPhoneDB db not present",
)
def test_within_species_mode(tmp_path):
    """Single-species analysis (standard CellPhoneDB scope) + structural-filter-ready hits."""
    from species_cpdb.database import build_cross_species_db
    from species_cpdb.synthetic import make_synthetic
    from species_cpdb.run import analyze_spatial

    db = build_cross_species_db(CPDB_DIR, str(tmp_path / "xdb"), species=["human"])
    assert db["cross_species_interactions"] == 0 and db["interactions"] > 100   # full human interactome

    adata = make_synthetic(seed=0, grid=22)
    human = adata[adata.obs["species"] == "human"].copy()            # single-species units
    res = analyze_spatial(human, db_zip=db["db_zip"], out_dir=str(tmp_path / "run"),
                          engine="neighborhood", radius=1.5, iterations=80, seed=1)
    within = res.cross_species_hits(pvalue=0.05, cross_only=False)   # within-species hits
    assert len(within) > 0 and (within["direction"] == "within-human").all()


def test_namespace_roundtrip():
    from species_cpdb import add_prefix, strip_prefix, species_of

    assert add_prefix("TGFB1", "human") == "hsa:TGFB1"
    assert strip_prefix("rno:Tgfb1") == "Tgfb1"
    assert species_of("hsa:VEGFA") == "human"
    assert species_of("rno:Vegfa") == "rat"


def _as_bin2cell(adata, microns_per_pixel=2.0):
    """Re-shape a synthetic section into exactly what ``b2c.bin_to_cell`` returns:
    bare gene symbols, summed counts (sparse CSR), obs_names = label ids as strings,
    obs bin_count/object_id/array_row/array_col/labels_joint_source, obsm['spatial']
    in full-res pixels and microns_per_pixel in uns['spatial'][lib]['scalefactors']."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from species_cpdb._namespace import species_of, strip_prefix

    counts = adata.layers["counts"]
    symbols = sorted({strip_prefix(v) for v in adata.var_names})
    pos = {s: i for i, s in enumerate(symbols)}
    X = np.zeros((adata.n_obs, len(symbols)))
    cell_sp = adata.obs["species"].astype(str).values
    for j, v in enumerate(adata.var_names):
        own = cell_sp == species_of(v)
        X[own, pos[strip_prefix(v)]] += counts[own, j]
    n = adata.n_obs
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({
        "object_id": np.arange(1, n + 1),
        "bin_count": rng.integers(5, 40, n),
        "array_row": adata.obsm["spatial"][:, 1] * 5.0,
        "array_col": adata.obsm["spatial"][:, 0] * 5.0,
        "labels_joint_source": np.where(np.arange(n) % 10 == 0, "secondary", "primary"),
        "species": cell_sp,
        "cell_type": adata.obs["cell_type"].astype(str).values,
        "region": adata.obs["region"].astype(str).values,
    }, index=[str(i) for i in range(1, n + 1)])
    cdata = ad.AnnData(sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=symbols))
    # synthetic grid step = 10 um -> pixels
    cdata.obsm["spatial"] = adata.obsm["spatial"] * 10.0 / microns_per_pixel
    cdata.uns["spatial"] = {"lib": {"scalefactors": {
        "microns_per_pixel": microns_per_pixel, "spot_diameter_fullres": 7.0}}}
    return cdata


def test_from_bin2cell_format():
    """bin2cell cell-level output -> units: um coords, prefixed genes, filters."""
    import numpy as np
    from species_cpdb.synthetic import make_synthetic
    from species_cpdb.preprocess import from_bin2cell

    cdata = _as_bin2cell(make_synthetic(seed=0, grid=12))
    units = from_bin2cell(cdata, celltype_key="cell_type", species="species", carry_obs=["region"])
    assert units.uns["spatial_units"] == "um"
    # pixels * microns_per_pixel: neighbouring grid cells are 10 um apart
    d = np.linalg.norm(units.obsm["spatial"][1] - units.obsm["spatial"][0])
    assert np.isclose(d, 10.0)
    assert all(":" in v for v in units.var_names)
    rat = units.obs["species"] == "rat"
    hsa = [i for i, v in enumerate(units.var_names) if v.startswith("hsa:")]
    assert units.X[np.where(rat)[0]][:, hsa].nnz == 0          # rat cells: no human genes
    tot = np.asarray(units.X.sum(1)).ravel()
    assert np.allclose(tot[tot > 0], 1e4)                       # total-count normalised
    assert set(units.obs["label"]) >= {"rat_fibroblast", "human_cardiomyocyte"}
    assert "region" in units.obs

    only_primary = from_bin2cell(cdata, celltype_key="cell_type", species="species",
                                 labels_source=["primary"], min_bins=10)
    assert only_primary.n_obs < units.n_obs
    one_sp = from_bin2cell(cdata, celltype_key="cell_type", species="human",
                           keep_genes={"hsa:TGFB1", "hsa:TGFBR2"})
    assert list(one_sp.var_names) == ["hsa:TGFB1", "hsa:TGFBR2"]


def test_bin2cell_end_to_end(tmp_path):
    """bin2cell cells through the neighborhood engine recover the planted signal."""
    from species_cpdb.database import build_cross_species_db
    from species_cpdb.synthetic import make_synthetic
    from species_cpdb.preprocess import prepare_units
    from species_cpdb.run import analyze_spatial

    db = build_cross_species_db(CPDB_DIR, str(tmp_path / "xdb"))
    cdata = _as_bin2cell(make_synthetic(seed=0, grid=22))
    units = prepare_units("visium_hd", cdata=cdata, celltype_key="cell_type", species="species")
    res = analyze_spatial(units, db_zip=db["db_zip"], out_dir=str(tmp_path / "run"),
                          engine="neighborhood", radius=15.0, iterations=100, seed=1)
    hits = res.cross_species_hits(pvalue=0.05)
    assert hits["interacting_pair"].str.contains("Tgfb").any()


def _seed_sequence_cache(cache_dir):
    import json, os
    os.makedirs(cache_dir, exist_ok=True)
    recs = {
        "ssc:TGFB1": {"accession": "P07200", "sequence": "SSSSMAAA", "signal_end": 4, "reviewed": True, "gene": "TGFB1"},
        "hsa:TGFBR1": {"accession": "P36897", "sequence": "MBBB", "signal_end": 0, "reviewed": True, "gene": "TGFBR1"},
        "hsa:TGFBR2": {"accession": "P37173", "sequence": "MCCC", "signal_end": 0, "reviewed": True, "gene": "TGFBR2"},
    }
    with open(os.path.join(cache_dir, "uniprot_sequences.json"), "w") as fh:
        json.dump(recs, fh)


def test_fetch_sequences_uses_cache(tmp_path, monkeypatch):
    """Cached genes never hit the network; signal peptide stripped; misses warned."""
    import pytest as _pt
    from species_cpdb import sequences

    _seed_sequence_cache(str(tmp_path))
    monkeypatch.setattr(sequences, "lookup_uniprot", lambda *a, **k: None)  # offline
    with _pt.warns(UserWarning, match="no UniProt sequence"):
        seqs = sequences.fetch_sequences(["ssc:TGFB1", "hsa:TGFBR2", "rno:Nope"], cache_dir=str(tmp_path))
    assert seqs == {"ssc:TGFB1": "MAAA", "hsa:TGFBR2": "MCCC"}
    full = sequences.fetch_sequences(["ssc:TGFB1"], cache_dir=str(tmp_path), strip_signal_peptide=False)
    assert full["ssc:TGFB1"] == "SSSSMAAA"
    prov = sequences.sequence_provenance(["ssc:TGFB1"], cache_dir=str(tmp_path))
    assert prov.loc[0, "accession"] == "P07200"


def test_run_structural_filter_autofetch(tmp_path):
    """No sequences= -> fetched (here from cache); complex names resolved via interactors."""
    import pandas as pd
    from species_cpdb.structure import run_structural_filter, MockFoldingBackend

    _seed_sequence_cache(str(tmp_path / "cache"))
    hits = pd.DataFrame({  # celltype-engine style: receptor named as a complex
        "interacting_pair": ["ssc:TGFB1_hsa:TGFbeta_receptor1"],
        "interactors": ["ssc:TGFB1->hsa:TGFBR1+hsa:TGFBR2"],
        "cross_species": [True],
    })
    backend = MockFoldingBackend(default={"iptm": 0.8})
    filtered, scores = run_structural_filter(
        hits, out_dir=str(tmp_path / "sf"), backend=backend, sequence_cache_dir=str(tmp_path / "cache"),
        run_prodigy=False, min_pdockq=None, max_interface_pae=None)
    assert len(scores) == 1 and len(filtered) == 1


def test_gpu_check_refuses_without_gpu(tmp_path, monkeypatch):
    """device='gpu' fails fast (before fetching/folding) when no GPU is visible."""
    import pandas as pd
    import pytest as _pt
    from species_cpdb import structure

    calls = []

    class Recording(structure.FoldingBackend):  # real (non-mock) device semantics
        def fold(self, name, chains, out_dir, device="gpu"):
            calls.append(device)
            return structure.FoldResult(interacting_pair=name, iptm=0.9)

    hits = pd.DataFrame({"interacting_pair": ["ssc:IL6_hsa:IL6R"], "cross_species": [True]})
    seqs = {"ssc:IL6": "MA", "hsa:IL6R": "MB"}
    kw = dict(out_dir=str(tmp_path), backend=Recording(), run_prodigy=False,
              min_pdockq=None, max_interface_pae=None)

    monkeypatch.setattr(structure, "gpu_available", lambda: False)
    with _pt.raises(RuntimeError, match="no NVIDIA GPU"):
        structure.run_structural_filter(hits, seqs, **kw)            # default device='gpu'
    assert calls == []                                               # nothing folded
    with _pt.raises(ValueError):
        structure.run_structural_filter(hits, seqs, device="auto", **kw)
    structure.run_structural_filter(hits, seqs, device="cpu", **kw)  # explicit CPU opt-in
    monkeypatch.setattr(structure, "gpu_available", lambda: True)
    structure.run_structural_filter(hits, seqs, **kw)
    assert calls == ["cpu", "gpu"]


def test_gpu_available_respects_hidden_devices(monkeypatch):
    from species_cpdb import structure

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert structure.gpu_available() is False


_FAKE_COLABFOLD = '''#!{python}
import json, os, sys, time
fasta, out = sys.argv[-2], sys.argv[-1]
if os.environ.get("FAKE_FAIL"):
    print("ERROR: MSA server rate limit", file=sys.stderr)
    sys.exit(3)
os.makedirs(out, exist_ok=True)
print("Query 1/1: test (length 4)", flush=True)
for p in (0, 50, 100):
    sys.stderr.write(f"\\rCOMPLETE: {{p}}%")
    sys.stderr.flush()
print("\\nalphafold2_multimer_v3_model_1 recycle=3 pLDDT=88 pTM=0.8 ipTM=0.77", flush=True)
json.dump({{"iptm": 0.77, "ptm": 0.8, "pae": [[1, 9, 9], [9, 1, 9], [9, 9, 1]]}},
          open(os.path.join(out, "x_scores_rank_001_model_1.json"), "w"))
open(os.path.join(out, "x_unrelaxed_rank_001_model_1.pdb"), "w").write("END\\n")
'''


def _fake_colabfold(tmp_path):
    import sys

    exe = tmp_path / "bin" / "colabfold_batch"
    exe.parent.mkdir()
    exe.write_text(_FAKE_COLABFOLD.format(python=sys.executable))
    exe.chmod(0o755)
    return str(exe)


def test_colabfold_streams_progress(tmp_path, capsys):
    """ColabFold's log reaches the notebook/terminal live, plus per-fold progress lines."""
    import pandas as pd
    from species_cpdb import structure

    backend = structure.ColabFoldBackend(executable=_fake_colabfold(tmp_path))
    hits = pd.DataFrame({"interacting_pair": ["ssc:IL6_hsa:IL6R"], "cross_species": [True]})
    _, scores = structure.run_structural_filter(
        hits, {"ssc:IL6": "MA", "hsa:IL6R": "MBC"}, out_dir=str(tmp_path / "s"),
        backend=backend, device="cpu", run_prodigy=False,
        min_pdockq=None, max_interface_pae=None)
    out = capsys.readouterr().out
    assert "[1/1] folding ssc:IL6_hsa:IL6R -- 2 chains, 5 aa" in out
    assert "COMPLETE: 100%" in out and "ipTM=0.77" in out       # ColabFold's own log
    assert "[1/1] done in" in out and "folded 1/1" in out
    assert scores.iptm.iloc[0] == 0.77
    logs = list((tmp_path / "s").rglob("colabfold_output.log"))
    assert logs and "Query 1/1" in logs[0].read_text()

    # quiet mode: no echo, but the log file is still written
    backend.verbose = False
    structure.run_structural_filter(
        hits, {"ssc:IL6": "MA", "hsa:IL6R": "MBC"}, out_dir=str(tmp_path / "q"),
        backend=backend, device="cpu", run_prodigy=False, progress=False,
        min_pdockq=None, max_interface_pae=None)
    assert capsys.readouterr().out == ""
    assert list((tmp_path / "q").rglob("colabfold_output.log"))


def test_colabfold_failure_shows_tail(tmp_path, monkeypatch):
    import pytest as _pt
    from species_cpdb import structure

    monkeypatch.setenv("FAKE_FAIL", "1")
    backend = structure.ColabFoldBackend(executable=_fake_colabfold(tmp_path), verbose=False)
    with _pt.raises(RuntimeError, match="(?s)exited with code 3.*MSA server rate limit"):
        backend.fold("a_b", [("a", "MA"), ("b", "MB")], str(tmp_path / "f"), device="cpu")
