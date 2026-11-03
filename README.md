# species_cpdb

**Spatial nearby-cell interaction inference**, wrapping the real
[CellPhoneDB](https://github.com/ventolab/CellphoneDB) v5 — with its flagship,
novel use case being **cross-species (e.g. rat ↔ human) prediction in xenografts**.

The core idea: a ligand-receptor prediction is only meaningful for cells that
are actually **near each other**. Dissociated scRNA-seq CellPhoneDB can't tell
whether a predicted hepatocyte-receptor / fibroblast-ligand pair is even
spatially feasible. Spatial data lets us gate predictions by proximity, and
(optionally) by physical bindability. Three orthogonal axes of evidence:

> **co-expression** (CellPhoneDB) × **spatial proximity** (the gate) × **physical bindability** (orthology → interface → structure)

For the cross-species case there is no ground-truth interactome, so the tool
**predicts** pairs from orthology and attaches an explicit confidence to each —
it does not pretend to look them up. In the xenograft: do human graft cells talk
to neighbouring rat host cells across the border, and via which L-R pairs?

## Species (any, not just rat)

Human is the reference (CellPhoneDB is human); every other species is defined by
its human ortholog map, so you pick which species to build for:

```python
build_cross_species_db("data/cpdb", "data/xdb", species=["human", "mouse"])
build_cross_species_db("data/cpdb", "data/xdb", species=["human", "pig"])
build_cross_species_db("data/cpdb", "data/xdb", species=["human", "pig", "rhesus"])  # >2 is fine
```

Registered species (`species_cpdb.SPECIES`, KEGG-style prefixes): human `hsa`,
mouse `mmu`, rat `rno`, pig `ssc`, rhesus `mcc`, cynomolgus `mcf`, marmoset
`cja`, rabbit `ocu`, dog `cfa`, sheep `oas`, cow `bta`, zebrafish `dre`. Add your
own with `species_cpdb.register(Species(...))`. With N species the builder emits
every `S1:ligand → S2:receptor` variant (including non-human↔non-human, bridged
through the shared human ortholog).

**Real orthologs are bundled.** The package ships a real Ensembl ortholog table
(`load_orthologs(source="ensembl")`) covering the whole CellPhoneDB interactome
across *every* supported species — because human is the reference, one
human→all-species table serves every combination (no N² pairwise files). It's the
default for `build-db`; just pick your species:

```bash
species-cpdb build-db --cpdb-dir data/cpdb --species human pig --out data/xdb   # uses real orthologs
```

To refresh it (Ensembl updates releases) or add a species, re-fetch once:

```bash
species-cpdb fetch-orthologs --cpdb-dir data/cpdb --species all --out data/orthologs.csv
```

*(Provenance: the bundled table was fetched from Ensembl BioMart (`www.ensembl.org`,
current release) over the CellPhoneDB v5 interactome; 16,685 rows, 11 species.
`% identity` is target-to-human protein identity; re-fetch to pin a release.)*

## Single-species mode

Set `species=["human"]` (or any one species) to run **ordinary spatial
CellPhoneDB** — the full within-species interactome — but with this tool's extras
that standard CellPhoneDB lacks: the per-spot `neighborhood` engine and, above
all, the **AlphaFold/PRODIGY structural filter** on the results. Read hits with
`cross_species_hits(cross_only=False)` and filter them with
`run_structural_filter(..., cross_only=False)`.

## Platforms (the analysis unit differs)

Pass `platform=` — it changes what "a cell" is and how proximity is measured:

| platform | unit | how |
|---|---|---|
| `visium_v1` | (spot × cell type) **pseudo-cells** | deconvolve each spot's observed expression by `abundance × reference_signature` (from cell2location / Tangram). Within-spot cell types are co-located; neighbouring spots drive between-spot interactions. |
| `xenium` / `merfish` / `cosmx` | one segmented **cell** | no deconvolution; neighbour graph on cells |
| `visium_hd` | **bin2cell** segmented **cell** | 2 µm bins grouped into cells by [bin2cell](https://github.com/Teichlab/bin2cell) (`b2c.bin_to_cell`); you annotate cell types; coordinates converted to **µm**. Or 8/16 µm bins + cell2location → pseudo-cells, as for v1 |

See `species_cpdb.preprocess.prepare_units`.

## Scope (which cells enter the mean)

CellPhoneDB's per-cell-type mean is global by default. Restrict it first:

- `scope="global"` — whole tissue.
- `scope="region"` — your own spatial annotations (`obs['region']`, e.g.
  `graft_border`, `host_distal`); the `human_fibroblast` mean becomes
  *border-local*.
- `scope="window"` — a disk of radius *r* around a chosen spot ("spot g, radius 6").

## Engines (cell-type vs per-spot adjacency)

`analyze_spatial(..., engine=...)`:

- `engine="celltype"` (default) — wraps **real CellPhoneDB**: cell-type mean
  expression + label permutation, with spatial microenvironments gating which
  cell-type pairs are eligible. Returns a `CpdbResult`.
- `engine="neighborhood"` — a native **per-spot adjacency** test: for
  `ligand(S1) → receptor(S2)` it scores the mean of `ligand[i]·receptor[j]` over
  edges where an S1-cell *i* is within `radius` of an S2-cell *j*, with a
  permutation null that shuffles expression *within each species*. A low p-value
  means the ligand-high and receptor-high cells are genuinely **adjacent**, not
  merely both present in the niche. Returns a `NeighborhoodResult`. This is the
  finer-resolution view (it typically keeps fewer cell-type-pair columns — only
  the ones that actually touch).

Both return an object with `.cross_species_hits(pvalue=…)` and both feed
`plotting.dotplot(...)` unchanged.

## Pluggable reliability

Each axis is an argument, so you choose how strict to be.

**Confidence** (how the cross-species prior is scored) — on `build_cross_species_db`:

```python
build_cross_species_db("data/cpdb", "data/xdb", confidence="orthology")  # whole-protein % identity
build_cross_species_db("data/cpdb", "data/xdb", confidence="interface")  # binding-residue conservation,
                                                                         # falls back to orthology per gene
```

`interface` records provenance per pair (`confidence_method`, e.g.
`interface(2 iface, 1 fallback)`). New scorers plug in via
`species_cpdb.confidence.ConfidenceModel`.

For **real** interface data (per-species binding-residue conservation), build a
table with `species_cpdb.interface` and pass it in:

```python
from species_cpdb.interface import build_interface_table
# providers: sequence_provider(gene, species)->seq (UniProt); interface_provider(gene)->[positions]
iface = build_interface_table(genes, ["human", "pig"], sequence_provider, interface_provider)
build_cross_species_db(..., confidence="interface", interface_table=iface)
```

`build_interface_table` aligns each species' ortholog to the human protein and
scores % identity *at the interface residues only* (`interface_conservation`).
Interface residues come from solved complex structures (PDBe/PISA/SIFTS) or from
an AlphaFold model of the complex — see the `interface.py` fetchers. The foreign
partner's interface conservation drives the pair's confidence; the native (human)
partner doesn't, and genes without interface data fall back to orthology.

**Ligand-class-aware proximity** — secreted ligands act at a diffusion radius,
membrane/juxtacrine only at contact:

```python
res.cross_species_hits(ligand_class_proximity=True,
                       radius_secreted=2.0, radius_contact=1.0)
```

Uses CellPhoneDB's `secreted` flag: a non-secreted pair is dropped unless its two
cell types are actually within `radius_contact`.

**Structural viability** (AlphaFold-Multimer / AF3 + PRODIGY) — a *downstream*
filter on the shortlist, so you fold dozens of survivors, never the whole DB.
Two ways:

*The tool runs it for you* (fold → ipTM/PAE → PRODIGY ΔG → your cutoffs), with
the backend using your GPU like cell2location:

```python
from species_cpdb.structure import run_structural_filter, ColabFoldBackend
kept, scores = run_structural_filter(
    hits,                                  # sequences fetched from UniProt automatically
    out_dir="struct/", backend=ColabFoldBackend(), device="gpu",    # default; errors without a GPU
    min_iptm=0.5, max_interface_pae=15, max_delta_g=-6.0)           # your AF + affinity cutoffs
```

**Sequences are fetched for you.** Each chain of each shortlisted complex is
looked up on UniProt by gene symbol + organism (so `rno:Tgfb1` gets the *rat*
protein), preferring reviewed Swiss-Prot entries and falling back to unreviewed
TrEMBL (flagged). Sequences are cached in `~/.cache/species_cpdb` (or
`sequence_cache_dir=`), so each gene is downloaded once per machine; signal
peptides are stripped. `species_cpdb.sequences.sequence_provenance(genes)` lists
the accession used for each. Pass your own `sequences={'ssc:TGFB1': 'MA...'}` to
override. Pro-domains (e.g. the TGF-β latency peptide) are *not* trimmed — check
those by hand for complexes you care about.

The folding backend (localcolabfold / Boltz) is an **opt-in** install needed
*only* when you call this; PRODIGY (`pip install .[structure]`) is CPU and light.
A `MockFoldingBackend` and `CommandFoldingBackend` (wire Boltz/Chai/AF3) are
provided.

**GPU check.** `device="gpu"` is the default and is checked *before* anything is
fetched or folded: if no NVIDIA GPU is visible (`nvidia-smi`), or ColabFold's
JAX isn't the CUDA build, it stops with an error saying how to get one (e.g. a
Colab GPU runtime) instead of silently folding on CPU for hours. Pass
`device="cpu"` only if you really mean to fold on CPU. Apple-silicon Macs have no
CUDA GPU, so fold on Colab / a cluster / a cloud GPU.

**Progress.** Folding is slow (minutes to well over ten per complex on a free
Colab T4, mostly queueing at the public MSA server), so you can see it working:
each complex prints a start line (`[2/5] folding ... -- 3 chains, 1012 aa`) and a
finish line (time taken, ipTM, interface PAE, ΔG), and ColabFold's own log (MSA
queue, model progress, per-model pLDDT/ipTM) streams live into the notebook cell.
The log is also saved as `colabfold_output.log` in each fold's folder, and if a
fold fails its last lines are shown in the error. Turn the lines off with
`progress=False` and the live log with `ColabFoldBackend(verbose=False)`.

**On Google Colab**, install with Colab's own pandas pinned. The newest
`anndata`/`scanpy` need pandas ≥ 2.3, which would otherwise replace the pandas
Colab ships with and break the `google-colab` package:

```python
!pip install -q "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold"
!pip install -q "species_cpdb[structure]" "pandas==$(python -c 'import pandas; print(pandas.__version__)')"
```

*Or fold elsewhere and feed results in*:

```python
from species_cpdb.structure import filter_hits_structurally, write_fold_inputs
write_fold_inputs(hits, out_dir="fastas/")   # ready-to-fold multimer FASTAs (UniProt sequences)
kept = filter_hits_structurally(hits, scores=my_scores, min_iptm=0.5, max_delta_g=-6.0)
```

Note: AlphaFold predicts *structure* (ipTM/PAE), not affinity — PRODIGY/FoldX give
the ΔG for `max_delta_g`. Complexes are folded subunit by subunit: celltype-engine
hits name complexes (`hsa:TGFbeta_receptor1`), so their member genes are read from
the `interactors` column (`ssc:TGFB1->hsa:TGFBR1+hsa:TGFBR2`).

## The idea in one picture

```
CellPhoneDB human interactome        human<->rat orthologs (Ensembl)
              |                                 |
              +------------------+--------------+
                                 v
           cross-species CellPhoneDB database        (database.py)
             hsa:L - hsa:R   (human, conf 1.0)
             rno:L - rno:R   (rat)
             rno:L - hsa:R   (rat ligand  -> human receptor)   <-- the point
             hsa:L - rno:R   (human ligand -> rat receptor)    <-- the point
                                 |
   spatial neighbour graph ------+                              (spatial.py)
   -> niches -> microenvironments (only co-located cell types tested)
                                 v
            real CellPhoneDB statistical method                 (run.py)
                                 v
          annotated cross-species hits + dotplots               (plotting.py)
```

**Key trick:** both species live in one gene namespace (`hsa:` / `rno:`). A
merged counts matrix holds the union of prefixed genes; a human spot expresses
only `hsa:` genes (its `rno:` entries are 0) and vice versa. A cross-species
interaction is then just an ordinary CellPhoneDB row whose two partners carry
different prefixes — so CellPhoneDB runs unmodified.

**Confidence:** each cross-species pair is scored in `0..1` from the sequence
conservation (protein % identity) / ortholog quality of the *foreign*
(substituted) partner — a proxy for "will the native receptor still recognise
the foreign ligand". It rides along in `interaction_meta.csv` and is shown on
the dotplots (marker edge); CellPhoneDB itself treats every row equally.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install .              # add .[orthology] for the Ensembl fetch
# for dev: `pip install -e . --config-settings editable_mode=compat`
```

## Quickstart (synthetic, offline)

```bash
python examples/quickstart.py
```

Runs the full pipeline on a **human-graft-in-pig** Visium v1 section: builds the
human↔pig database, simulates mixed spots, deconvolves them to per-cell-type
pseudo-cells, runs CellPhoneDB with spatial microenvironments, and writes
`examples/out/dotplot_pig_to_human.png`, `dotplot_human_to_pig.png`, and
`cross_species_hits_pig.csv`, plus ready-to-fold multimer FASTAs in
`examples/out/fold_inputs/` (from bundled real UniProt example sequences, so it
stays offline). On the synthetic data the strongest cross-border
hit is pig-fibroblast **TGFB1/TGFB2 → human TGFβ-receptor** — the fibrotic signal
planted into the graft. Set `HOST` at the top of the script to `rat`, `mouse`,
`rhesus`, … to switch host species.

## On your real data

1. **Orthologs (real).** The bundled seed (rat/mouse/pig/rhesus) is *illustrative*
   for the demo. Fetch the real map for your host species:
   ```bash
   species-cpdb fetch-orthologs --species mouse --out data/orthologs_ensembl.csv
   ```
2. **Build the cross-species database:**
   ```bash
   species-cpdb build-db --cpdb-dir data/cpdb --orthologs data/orthologs_ensembl.csv --out data/xspecies_db
   ```
   (`data/cpdb` is a released CellPhoneDB db; download once with
   `python -c "from cellphonedb.utils import db_utils; db_utils.download_database('data/cpdb','v5.0.0')"`.)
3. **Run** — `--platform` prepares the analysis units for you:
   - **Visium v1** (deconvolve in-CLI from your cell2location outputs):
     ```bash
     species-cpdb run --platform visium_v1 --counts spots.h5ad \
         --abundance abundances.csv --signature reference_signature.csv \
         --db data/xspecies_db/cellphonedb.zip --out results/ --engine neighborhood --plot
     ```
     (`abundances.csv` = spot × cell-type; `reference_signature.csv` = cell-type ×
     gene; cell-type names like `human_fibroblast` give the species, else pass
     `--celltype-species`.)
   - **Xenium / MERFISH / CosMx** (single cell):
     ```bash
     species-cpdb run --platform xenium --counts cells.h5ad \
         --celltype-key cell_type --species-key species \
         --db data/xspecies_db/cellphonedb.zip --out results/ --plot
     ```
   - **Visium HD via bin2cell** (segmented cells): pass the *cell-level* AnnData
     from `b2c.bin_to_cell`, after annotating a cell-type column on it:
     ```bash
     species-cpdb run --platform visium_hd --counts b2c_cells.h5ad \
         --celltype-key cell_type --species-key species \
         --db data/xspecies_db/cellphonedb.zip --out results/ --engine neighborhood --radius 30 --plot
     ```
     Coordinates are converted from image pixels to µm (Space Ranger's
     `microns_per_pixel`), so `--radius` is in **µm** (default 30). Raw or destriped
     counts are total-count normalised (skipped if scanpy marked the data `log1p`;
     `--layer counts` to use a raw layer). `--hd-species human` for a
     single-species section; `--min-bins` drops tiny objects; `--labels-source
     primary` keeps only H&E-segmented cells. Only database genes are kept, to fit
     a whole-transcriptome section in memory. Python: `from_bin2cell(cdata, ...)`.
   - **Already prepared** (units AnnData with prefixed genes + `obs['label']`):
     `species-cpdb run --counts units.h5ad --db …/cellphonedb.zip --out results/`

   Add `--scope region --region graft_border` to localise, or `--engine neighborhood`
   for the per-spot adjacency test.

## Modules

| module | role |
|---|---|
| `species.py`   | registry of supported species (prefixes, Ensembl homolog stems) |
| `orthology.py` | human→species orthologs from Ensembl BioMart (any species); counts merge |
| `database.py`  | expand the human interactome into a cross-species CellPhoneDB db |
| `confidence.py`| pluggable cross-species confidence models (`orthology`, `interface`, …) |
| `interface.py` | binding-interface conservation: align ortholog, score identity at interface residues |
| `structure.py` | downstream AlphaFold viability filter + `write_fold_inputs` (multimer FASTAs) |
| `preprocess.py`| platform → analysis units: v1 deconvolution to pseudo-cells, bin2cell (Visium HD) cells, single-cell passthrough |
| `sequences.py` | UniProt sequence lookup per species (gene symbol + taxon), local cache, bundled demo set |
| `synthetic.py` | Visium-like xenograft simulators (spot-level and v1-mixture) for the demo |
| `spatial.py`   | neighbour graph → niches → microenvironments; region/window scope; ligand-class proximity |
| `run.py`       | call the real CellPhoneDB method; `analyze_spatial` orchestrator; annotate hits |
| `neighborhood.py` | native per-spot adjacency L-R engine (`engine="neighborhood"`) |
| `plotting.py`  | CellPhoneDB-style dotplots of cross-species interactions |

## Caveats (this is a prediction tool)

- Cross-species pairs are inferred from orthology, **not** measured binding.
  Treat confidence as a prior, not a probability of interaction.
- Protein annotations (secreted / receptor) are assumed conserved across
  orthologs; a within-rat interaction is assumed valid when both orthologs exist.
- Visium spots are mixtures; cell-type resolution is only as good as your
  cell2location deconvolution. Microenvironments make the test *spatial* but it
  remains cell-type-level, like CellPhoneDB itself.
- Visium HD is probe-based and 10x ships human and mouse probe sets only: in a
  human-in-rat graft, check how rat transcripts are captured (and how each cell's
  species is called) before trusting cross-species calls on HD.
- Genes with no rat ortholog cannot form cross-species pairs by construction —
  some human ligands/receptors are simply untransferable, which the tool makes
  explicit rather than hiding.
