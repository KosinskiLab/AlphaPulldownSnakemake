# AlphaPulldownSnakemake

AlphaPulldownSnakemake provides a convenient way to run AlphaPulldown using a Snakemake pipeline. This lets you focus entirely on **what** you want to compute, rather than **how** to manage dependencies, versioning, and cluster execution.  

**Helpful links:** [AlphaPulldown documentation](https://github.com/KosinskiLab/AlphaPulldown/wiki) · [Precalculated feature databases](https://github.com/KosinskiLab/AlphaPulldown/wiki/Features-Database) · [Downstream analysis guide](https://github.com/KosinskiLab/AlphaPulldown/wiki/Downstream-Analysis)

## 1. Installation
Create and activate the conda environment:

```bash
conda env create \
  -n snake \
  -f https://raw.githubusercontent.com/KosinskiLab/AlphaPulldownSnakemake/2.4.0/workflow/envs/alphapulldown.yaml
conda activate snake
```

This environment file installs Snakemake and all required plugins via conda and pulls in `alphapulldown-input-parser>=0.4.0` from PyPI in a single step.

That's it, you're done!

## 2. Configuration

### Create a working directory

Create a new processing directory for your project:

```bash
snakedeploy deploy-workflow \
  https://github.com/KosinskiLab/AlphaPulldownSnakemake \
  AlphaPulldownSnakemake \
  --tag 2.4.0
cd AlphaPulldownSnakemake
```

### Setup protein folding jobs

Create or edit the sample sheet `config/sample_sheet.csv` listing the proteins you want to fold. The simplest format uses one folding specification per line, for example UniProt IDs:

```
P01258+P01579
P01258
P01579
```

Each line represents one folding job:
- `P01258+P01579` - fold these two proteins together as a complex
- `P01258` - fold this protein as a monomer
- `P01579` - fold this protein as a monomer

<details>
<summary>Advanced protein specification options</summary>

You can also specify:
- **FASTA file paths** instead of UniProt IDs: `/path/to/protein.fasta`
- **Specific residue regions**: `Q8I2G6:1-100` (residues 1-100 only)
- **Discontinuous regions**: `Q8I2G6:1-100:150-200` (two separate regions from the same protein)
- **Multiple copies**: `Q8I2G6:2` (dimer of the same protein)
- **Combinations**: `Q8I2G6:2:1-100+Q8I5K4` (dimer of residues 1-100 plus another protein)
- **Copies plus discontinuous regions**: `Q8I2G6:2:1-100:150-200+Q8I5K4`

The same copy/range syntax also works when the workflow generates AlphaFold 3
JSON features (`--data_pipeline: alphafold3`). Examples:

- `Q8I2G6_af3_input.json:1-100`
- `Q8I2G6_af3_input.json:1-100:150-200`
- `Q8I2G6_af3_input.json:2:1-100:150-200+Q8I5K4_af3_input.json`

In that mode the Snakefile rewrites logical inputs such as
`Q8I2G6:1-100:150-200` to the corresponding
`Q8I2G6_af3_input.json:1-100:150-200` feature reference automatically.
AlphaPulldown preserves those discontinuous regions as one gapped polymer
chain with preserved residue-number gaps.
This keeps retained fragments intra-chain, so template contacts between those
fragments are not masked as inter-chain interactions.
The original residue IDs are written to the mmCIF author-numbering fields
(`auth_seq_id` and `pdbx_PDB_ins_code`); overlapping IDs are disambiguated with
insertion codes such as `2A`, `2B`, and so on.
Make sure the prediction container or runtime environment includes a matching
AlphaPulldown build together with `alphapulldown-input-parser>=0.4.0`.

</details>

### Configure input files

Edit `config/config.yaml` and set the path to your sample sheet:

```yaml
input_files:
  - "config/sample_sheet.csv"
```

### Setup pulldown experiments

If you want to test which proteins from one group interact with proteins from another group, create a second file such as `config/baits.txt`:

```
Q8I2G6
```

And update your config:

```yaml
input_files:
  - "config/sample_sheet.csv"
  - "config/baits.txt"
```

This will test all combinations: every protein in `config/sample_sheet.csv` paired with every protein in `config/baits.txt`.

<details>
<summary>Multi-file pulldown experiments</summary>

You can extend this logic to create complex multi-partner interaction screens by adding more input files. For example, with three files:

```yaml
input_files:
  - "proteins_A.txt"  # 5 proteins
  - "proteins_B.txt"  # 3 proteins
  - "proteins_C.txt"  # 2 proteins
```

This will generate all possible combinations across the three groups, creating 5×3×2 = 30 different folding jobs. Each job will contain one protein from each file, allowing you to systematically explore higher-order protein complex formation.

**Note**: The number of combinations grows multiplicatively, so be mindful of computational costs with many files.

</details>

## 3. Execution

Run the pipeline locally:

```bash
snakemake --profile config/profiles/desktop --cores 8
```

<details>
<summary>Cluster execution</summary>

For running on a SLURM cluster, first create a virtual terminal e.g. using `screen`:

```bash
screen -S snakemake_session
```
Then activate your conda/mamba environment:
```bash
mamba activate snake
```
Finally, use the slurm executor plugin:
```bash
snakemake \
  --executor slurm \
  --profile config/profiles/slurm \
  --jobs 200 \
  --restart-times 5
```

Detach with `Ctrl + A` then `D`. Reattach later with `screen -r snakemake_session`.

Job specific logs are created automatically and stored in your `AlphaPulldownSnakemake/slurm_logs` directory.

</details>

## 4. Results

After completion, you'll find:
- **Predicted structures** in PDB/CIF format in the output directory
- **Per-fold interface scores** in `output/predictions/<fold>/interfaces.csv`
- **Aggregated interface summary** in `output/reports/all_interfaces.csv` when `generate_recursive_report: true`
- **Interactive APLit web viewer (recommended)** for browsing all jobs, PAE plots and AlphaJudge scores
- **Optional Jupyter notebook** with 3D visualizations and quality plots
- **Results table** with confidence scores and interaction metrics

# Recommended: explore results with APLit

[APLit](https://github.com/KosinskiLab/aplit)
 is a Streamlit-based UI for browsing AlphaPulldown runs (AF2 and AF3) and AlphaJudge metrics.

Install APLit (once):
```bash
pip install git+https://github.com/KosinskiLab/aplit.git
```

Then launch it from your project directory, pointing it to the predictions folder:
```bash
aplit --directory output/predictions
```

This starts a local web server (by default at `http://localhost:8501`) where you can:

- Filter and sort jobs by ipTM, PAE or AlphaJudge scores

- Inspect individual models in 3D (3Dmol.js)

- View PAE heatmaps and download structures / JSON files

On a cluster, run aplit on the login node and forward the port via SSH:
```bash
# on cluster
aplit --directory /path/to/project/output/predictions --no-browser
```
```bash
# on your laptop
ssh -N -L 8501:localhost:8501 user@cluster.example.org
```

Then open `http://localhost:8501` in your browser.


---

## Advanced Configuration

### SLURM defaults for structure inference
Override default values to match your cluster:

```yaml
slurm_partition: "gpu"                      # which partition/queue to submit to
slurm_qos: "normal"                         # optional QoS if your site uses it
structure_inference_gpus_per_task: 1        # number of GPUs each inference job needs
structure_inference_gpu_model: "3090"       # optional GPU model constraint (remove to allow any)
structure_inference_tasks_per_gpu: 0        # <=0 keeps --ntasks-per-gpu unset in the plugin
slurm_exclude_nodes: ""                     # optional comma-separated nodes to avoid (sbatch --exclude)
structure_inference_max_runtime: 10080      # cap wall time (min) at the partition MaxTime
```

`structure_inference_gpus_per_task` and `structure_inference_gpu_model` are read by the
Snakemake Slurm executor plugin and translated into `--gpus=<model>:<count>` (or `--gpus=<count>` if
no model is specified). We no longer use `slurm_gres`; requesting GPUs exclusively through these
fields keeps the job submission consistent across clusters.

`structure_inference_tasks_per_gpu` toggles whether the plugin also emits `--ntasks-per-gpu`. Leaving
the default `0` prevents that flag, which avoids conflicting with the Tres-per-task request on many
systems. Set it to a positive integer only if your site explicitly requires `--ntasks-per-gpu`.

The remaining optional fields help with two common cluster issues: keeping inference off GPUs it
can't use, and large complexes running out of GPU memory. Defaults are sensible; expand below only if
you hit these.

<details>
<summary>Avoiding unsuitable GPUs (<code>slurm_exclude_nodes</code>, <code>gpu_model</code>) and the runtime cap</summary>

- **Restrict to one model** with `structure_inference_gpu_model` (e.g. `"A100"`) → the plugin emits
  `--gpus=<model>:<count>`. Accepts a single model name; leave `""` for any.
- **Route by complex size (VRAM)** with `structure_inference_gpu_tiers` → list your GPU pool as
  tiers of `{min_vram_gb, nodes}`. A complex's estimated peak VRAM (≈ `per_token_sq·N²`) selects the
  smallest tier that fits and all *smaller*-GPU nodes are excluded, so the job runs on **any** GPU at
  or above that tier — using the whole pool, not one pinned model. A complex larger than every tier
  uses the biggest tier and spills to host RAM via unified memory.

  ```yaml
  # Example for EMBL gpu-el8 — replace nodes with your cluster's (nothing is hard-coded):
  structure_inference_gpu_vram_headroom: 1.0   # <1.0 tolerates that fraction of host spill
  structure_inference_gpu_tiers:
    - {min_vram_gb: 24, nodes: "gpu21,gpu22,gpu29,gpu30,gpu31,gpu32,gpu33,gpu34,gpu35,gpu36,gpu37"}
    - {min_vram_gb: 40, nodes: "gpu25,gpu26,gpu27,gpu28"}
    - {min_vram_gb: 48, nodes: "gpu40,gpu41,gpu42,gpu43,gpu44,gpu45,gpu46,gpu47,gpu48"}
    - {min_vram_gb: 80, nodes: "gpu38,gpu39"}
  ```

  When set this drives `--exclude` per job and **overrides** `structure_inference_gpu_model` (the two
  would conflict). It's the practical "fit to GPU" lever: requested host RAM is a separate pool and
  does not size GPU VRAM, but excluding too-small GPUs by length does. Use explicit comma node lists
  (bracket ranges may be glob-expanded by the shell). Multi-partition routing (e.g. EMBL's bigger
  `gpu-training` cards) is out of scope — keep one partition and let unified memory spill the tail.
- **Exclude specific nodes** with `slurm_exclude_nodes` → passed verbatim to `sbatch --exclude`
  (e.g. `"gpu50,gpu51"`). Use it for nodes whose GPU the container can't use — e.g. a CUDA compute
  capability newer than the container's bundled `ptxas` (fails `ptxas too old` / `UNIMPLEMENTED`).
  `--exclude` is allowed in `slurm_extra` whereas `--constraint`/`--gres`/`--gpus` are not, so it is
  the supported way to drop a few nodes while keeping the rest of the partition.
- **`structure_inference_max_runtime`** caps per-job wall time (minutes). Wall time scales as
  `1440 * attempt`, so without a cap enough retries exceed the partition `MaxTime` and SLURM rejects
  the job with `Requested time limit is invalid`. Set it to your partition's `MaxTime`
  (`scontrol show partition <name>`); default 7 days (10080).

</details>

<details>
<summary>Unified memory for large complexes (<code>structure_inference_unified_memory</code>)</summary>

Large AlphaFold 3 inputs (or smaller-VRAM GPUs) can fail with `RESOURCE_EXHAUSTED` /
`Allocator (GPU_0_bfc) ran out of memory`. Inference enables JAX/XLA **unified (managed) memory** by
default so the model spills from GPU VRAM into host RAM instead of OOM-ing (slower while spilling, but
it completes) — the
[DeepMind-recommended setting](https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md)
for large inputs. It is exported inside the prediction container as:

```sh
export TF_FORCE_UNIFIED_MEMORY=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # don't grab a huge VRAM chunk up front
export XLA_CLIENT_MEM_FRACTION=$FRACTION      # how far past physical VRAM XLA may allocate
export XLA_PYTHON_CLIENT_MEM_FRACTION=$FRACTION
```

`XLA_PYTHON_CLIENT_PREALLOCATE=false` is required: without it XLA reserves a large
slice of VRAM immediately, which defeats the point of letting the allocator grow into
host RAM on demand.

```yaml
structure_inference_unified_memory: true     # set false to fail fast on OOM instead
structure_inference_xla_mem_fraction: auto   # "auto", or pin a number like 3.2
```

With the default `structure_inference_xla_mem_fraction: auto`, the fraction is computed
**per job at run time** as `(allocated host RAM) / (physical GPU VRAM)`: the GPU VRAM is
read with `nvidia-smi` once the job lands on a node, and the host RAM is the job's SLURM
`--mem` allocation (which scales with retry attempts). This keeps the unified-memory
ceiling within the SLURM allocation so XLA cannot oversubscribe host RAM beyond what the
job requested — which would otherwise get the job OOM-killed. The chosen fraction is
logged as a `[unified-memory]` line at the top of the job log. Pin a number instead if
you want a fixed multiplier regardless of GPU/RAM (mirrors the EMBL `run_AF_multimer.sh`
convention).

> The fraction is computed in the job shell rather than via the SLURM executor: the
> executor passes the submit environment through with `--export=ALL` but offers no
> per-job env hook, and the value depends on which GPU the job lands on (only known at
> run time). Computing it in the container shell also avoids the apptainer env-crossing
> that submit-side env vars would need.

Because spilling is slower, make sure the job also requests enough host RAM
(`structure_inference_ram_bytes`, in MB) to hold the overflow — under `auto` that RAM is
exactly what the fraction is sized against.

</details>

<details>
<summary>Length-aware memory requests (sized automatically from the input sequences)</summary>

Host RAM for both compute stages is requested **from the input sequence length**, so big
complexes get enough memory on the first attempt instead of failing and climbing the retry
ladder, while small jobs are not over-provisioned. The request is computed at scheduling
time by reading the per-chain FASTA(s) the pipeline already stages under
`<output_directory>/data/`:

```
create_features      mem = safety * (feature_create_ram_bytes + per_residue * seq_len)
structure_inference  mem = safety * (structure_inference_ram_bytes + per_token_sq * N^2)
```

- `seq_len` is the query length; `N` is the **total residues of the complex** (the
  AlphaFold token count, summed over chains and copy numbers). AlphaFold's pair
  representation is `O(N^2)`, hence the quadratic inference term.
- **The coefficients default by backend** (selected from `--data_pipeline` / `--fold_backend`).
  AlphaFold-Multimer (AF2) is heavier than AlphaFold 3 — measured AF2 inference host RSS was
  ~4× higher than AF3 at the same complex size, and AF2's feature stage runs HHblits (the
  main OOM source), whereas the AF3 pipeline is lighter. Defaults:

  | backend | feature base | feature /residue | inference base | inference /N² |
  |---|---|---|---|---|
  | `alphafold2` | 64000 MB | 40 MB | 24000 MB | 0.0055 |
  | `alphafold3` | 40000 MB | 25 MB | 16000 MB | 0.0045 |

  The AF3 inference quadratic is sized to the observed GPU-VRAM demand so that, with unified
  memory, the host spill ceiling (`host_mem / gpu_vram`) covers large complexes instead of
  OOM-ing.
- The first attempt already includes `mem_safety_factor` (default `1.25`) of head-room.
  **OOM retries still escalate** on top, multiplying by `..._ram_scaling ** (attempt - 1)`,
  so a bad estimate self-heals.
- Override any backend default by setting the matching key in `config/config.yaml`
  (`feature_create_ram_bytes`, `feature_create_ram_per_residue_mb`,
  `structure_inference_ram_bytes`, `structure_inference_ram_per_token_sq_mb`); an explicit
  value applies to all backends. Also tune `mem_safety_factor`, the `..._ram_scaling`
  factors, `structure_inference_runtime_minutes`, and `max_mem_mb` (set it to your largest
  node's RAM where an over-estimate would otherwise never schedule; `0` = no cap).
- The `..._ram_bytes` keys are the **fixed base** of each model rather than a flat request;
  raising a base only raises the floor. Setting `per_residue`/`per_token_sq` to `0`
  reproduces the old length-blind behaviour (a flat base × retry scaling).
- **Precomputed features:** when a chain is supplied via `feature_directory`, no
  `data/<chain>.fasta` is generated. Length is then recovered from the precomputed
  `<chain>_af3_input.json` (AF3) or from the parse-time length cache written by the length
  filter below (covers AF2 too). If neither is available the job falls back to the base
  allocation plus retry escalation. AF3 ligand atoms are not counted (no sequence), a small
  undercount absorbed by the safety margin.

</details>

<details>
<summary>Skipping over-large complexes (length filtering)</summary>

Folds that are too large to be worth submitting are **skipped before any job is created**,
so a single oversized complex (or one giant chain) doesn't waste a GPU/feature allocation
that will only OOM. Two configurable limits (in `config/config.yaml`):

```yaml
# Max TOTAL complex length (sum of all chains), per backend — selected by --fold_backend.
max_total_length_alphafold2: 5000     # AF2-Multimer
max_total_length_alphafold3: 7000     # AF3 handles larger inputs
# max_total_length: 6000              # optional single override for both backends
# Max length of any SINGLE protein; 0 = off (issue #33). A protein over this drops every
# fold containing it, so it is never even downloaded.
max_protein_length: 0
length_filter_fetch_uniprot: true     # set false for fully offline runs
```

- Lengths are resolved at **parse time** from, in order: a local FASTA, an
  already-downloaded `data/<id>.fasta`, the persistent cache
  `<output_directory>/.sequence_lengths.tsv`, and finally the UniProt REST API (cached for
  next time). Set a limit to `0` to disable it; if both are `0`, no resolution/fetching
  happens at all.
- Skipped folds are listed with reasons in `<output_directory>/skipped_folds.tsv` and logged
  as a `[length-filter]` warning. **Unknown lengths fail open** (the fold is kept), so a
  UniProt outage never silently drops work.
- First parse of a large all-UniProt sheet will fetch each unique length once (cached
  afterwards); already-downloaded inputs and local FASTAs are read without any network call.
- **Applies to every profile, including local/workstation runs** (it runs during workflow
  parsing, not in the executor). It's the only length-aware feature that does — the memory
  and GPU-routing settings are SLURM resources that local runs ignore. To attempt a complex
  larger than the caps on a big workstation, raise or zero the `max_total_length_*` values
  (and set `length_filter_fetch_uniprot: false` for offline use).

</details>

### Using precomputed features

If you have precomputed protein features, specify the directory:

```yaml
feature_directory:
  - "/path/to/directory/with/features/"
```

> **Note**: If your features are compressed, set `compress-features: True` in the config.

### Feature generation flags (`create_individual_features.py`)

You can tweak the feature-generation step by editing `create_feature_arguments` (or by running the
script manually). Commonly used flags:

- `--data_pipeline {alphafold2,alphafold3}` – choose the feature format to emit.
- `--db_preset {full_dbs,reduced_dbs}` – switch between the full BFD stack or the reduced databases.
- `--use_mmseqs2` – rely on the remote MMseqs2 API; skips local jackhmmer/HHsearch database lookups.
- `--use_precomputed_msas` / `--save_msa_files` – reuse stored MSAs or keep new ones for later runs.
- `--compress_features` – zip the generated `*.pkl` files (`.xz` extension) to save space.
- `--skip_existing` – leave existing feature files untouched (safe for reruns).
- `--seq_index N` – only process the N‑th sequence from the FASTA list.
- `--use_hhsearch`, `--re_search_templates_mmseqs2` – toggle template search implementations.
- `--path_to_mmt`, `--description_file`, `--multiple_mmts` – enable TrueMultimer CSV-driven feature sets.
- `--max_template_date YYYY-MM-DD` – required cutoff for template structures; keeps runs reproducible.


### Structure analysis & reporting

Post-inference analysis is enabled by default. You can disable it or add a project-wide summary in `config/config.yaml`:

```yaml
enable_structure_analysis: true             # skip alphaJudge if set to false
generate_recursive_report: true             # disable if you do not need all_interfaces.csv
recursive_report_arguments:                 # optional extra CLI flags for alphajudge
  --models_to_analyse: best
```

### Changing folding backends

To use AlphaFold3 or other backends:

```yaml
structure_inference_arguments:
  --fold_backend: alphafold3
  --<other-flags>
```

> **Note**: AlphaPulldown supports: `alphafold2`, `alphafold3`, and `alphalink` backends.

### Backend-specific flags

You can pass any backend CLI switches through `structure_inference_arguments`. Common options are listed below; keep or remove lines based on your needs.

<details>
<summary>AlphaFold2 flags</summary>

```yaml
structure_inference_arguments:
  --compress_result_pickles: False        # gzip AF2 result pickles
  --remove_result_pickles: False          # delete pickles after summary is created
  --models_to_relax: None                 # all | best | none
  --remove_keys_from_pickles: True        # strip large tensors from pickle outputs
  --convert_to_modelcif: True             # additionally write ModelCIF files
  --allow_resume: True                    # resume from partial runs
  --num_cycle: 3
  --num_predictions_per_model: 1
  --pair_msa: True
  --save_features_for_multimeric_object: False
  --skip_templates: False
  --msa_depth_scan: False
  --multimeric_template: False
  --model_names: None
  --msa_depth: None
  --description_file: None
  --path_to_mmt: None
  --desired_num_res: None
  --desired_num_msa: None
  --benchmark: False
  --model_preset: monomer
  --use_ap_style: False
  --use_gpu_relax: True
  --dropout: False
```
</details>

<details>
<summary>AlphaFold3 flags</summary>

```yaml
structure_inference_arguments:
  --jax_compilation_cache_dir: null
  --buckets: ['64','128','256','512','768','1024','1280','1536','2048','2560','3072','3584','4096','4608','5120']
  --flash_attention_implementation: triton
  --num_diffusion_samples: 5
  --num_seeds: null
  --debug_templates: False
  --debug_msas: False
  --num_recycles: 10
  --save_embeddings: False
  --save_distogram: False
```
</details>

### Database configuration

Set the paths to AlphaFold databases and backend weights:

```yaml
databases_directory: "/path/to/alphafold/databases"
backend_weights_directory: "/path/to/backend/weights"
```

---

## How to cite

If AlphaPulldown (or this workflow) contributed to your research, please cite [Molodenskiy et al., 2025](https://doi.org/10.1093/bioinformatics/btaf115):

```bibtex
@article{Molodenskiy2025AlphaPulldown2,
  author    = {Molodenskiy, Dmitry and Maurer, Valentin J. and Yu, Dingquan and
               Chojnowski, Grzegorz and Bienert, Stefan and Tauriello, Gerardo and
               Gilep, Konstantin and Schwede, Torsten and Kosinski, Jan},
  title     = {AlphaPulldown2—a general pipeline for high-throughput structural modeling},
  journal   = {Bioinformatics},
  volume    = {41},
  number    = {3},
  pages     = {btaf115},
  year      = {2025},
  doi       = {10.1093/bioinformatics/btaf115}
}
```
