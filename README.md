# AlphaPulldownSnakemake

AlphaPulldownSnakemake provides a convenient way to run AlphaPulldown using a Snakemake pipeline. This lets you focus entirely on **what** you want to compute, rather than **how** to manage dependencies, versioning, and cluster execution.  

**Helpful links:** [AlphaPulldown documentation](https://github.com/KosinskiLab/AlphaPulldown/wiki) · [Precalculated feature databases](https://github.com/KosinskiLab/AlphaPulldown/wiki/Features-Database) · [Downstream analysis guide](https://github.com/KosinskiLab/AlphaPulldown/wiki/Downstream-Analysis)

## 1. Installation
Create and activate the conda environment:

```bash
conda env create \
  -n snake \
  -f https://raw.githubusercontent.com/KosinskiLab/AlphaPulldownSnakemake/2.1.5/workflow/envs/alphapulldown.yaml
conda activate snake
```

This environment file installs Snakemake and all required plugins via conda and pulls in `alphapulldown-input-parser` from PyPI in a single step.

That's it, you're done!

## 2. Configuration

### Create a working directory

Create a new processing directory for your project:

```bash
snakedeploy deploy-workflow \
  https://github.com/KosinskiLab/AlphaPulldownSnakemake \
  AlphaPulldownSnakemake \
  --tag 2.1.6
cd AlphaPulldownSnakemake
```

### Setup protein folding jobs

Create a sample sheet `folds.txt` listing the proteins you want to fold. The simplest format uses UniProt IDs:

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
- **Multiple copies**: `Q8I2G6:2` (dimer of the same protein)
- **Combinations**: `Q8I2G6:2:1-100+Q8I5K4` (dimer of residues 1-100 plus another protein)

</details>

### Configure input files

Edit `config/config.yaml` and set the path to your sample sheet:

```yaml
input_files:
  - "folds.txt"
```

### Setup pulldown experiments

If you want to test which proteins from one group interact with proteins from another group, create a second file `baits.txt`:

```
Q8I2G6
```

And update your config:

```yaml
input_files:
  - "folds.txt"
  - "baits.txt"
```

This will test all combinations: every protein in `folds.txt` paired with every protein in `baits.txt`.

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

For running on a SLURM cluster, use the executor plugin:

```bash
screen -S snakemake_session
snakemake \
  --executor slurm \
  --profile config/profiles/slurm \
  --jobs 200 \
  --restart-times 5
```

Detach with `Ctrl + A` then `D`. Reattach later with `screen -r snakemake_session`.

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
```

`structure_inference_gpus_per_task` and `structure_inference_gpu_model` are read by the
Snakemake Slurm executor plugin and translated into `--gpus=<model>:<count>` (or `--gpus=<count>` if
no model is specified). We no longer use `slurm_gres`; requesting GPUs exclusively through these
fields keeps the job submission consistent across clusters.

`structure_inference_tasks_per_gpu` toggles whether the plugin also emits `--ntasks-per-gpu`. Leaving
the default `0` prevents that flag, which avoids conflicting with the Tres-per-task request on many
systems. Set it to a positive integer only if your site explicitly requires `--ntasks-per-gpu`.

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

