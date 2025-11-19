# AlphaPulldownSnakemake

AlphaPulldownSnakemake provides a convenient way to run AlphaPulldown using a Snakemake pipeline. This lets you focus entirely on **what** you want to compute, rather than **how** to manage dependencies, versioning, and cluster execution.

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
  --tag 2.1.5
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

### Using precomputed features

If you have precomputed protein features, specify the directory:

```yaml
feature_directory:
  - "/path/to/directory/with/features/"
```

> **Note**: If your features are compressed, set `compress-features: True` in the config.

### Structure analysis & reporting

Post-inference analysis is enabled by default. You can disable it or add a project-wide summary in `config/config.yaml`:

```yaml
enable_structure_analysis: true          # skip alphaJudge if set to false
generate_recursive_report: true          # set to false if you do not need all_interfaces.csv
recursive_report_arguments:              # optional extra CLI flags for alphajudge
  --models_to_analyse: best

# SLURM defaults (override to match your cluster)
slurm_partition: "gpu"
slurm_gres: "gpu:1"
slurm_qos: "normal"
```


### Changing folding backends

To use AlphaFold3 or other backends:

```yaml
structure_inference_arguments:
  --fold_backend: alphafold3
  --<other-flags>
```

> **Note**: AlphaPulldown supports: alphafold2, alphafold3 and alphalink backends.

### Database configuration

Set the path to your AlphaFold databases:

```yaml
databases_directory: "/path/to/alphafold/databases"
```

