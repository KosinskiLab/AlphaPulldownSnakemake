# AlphaPulldownSnakemake

A Snakemake pipeline for automated structure prediction using various backends (AlphaFold2, AlphaFold3, AlphaLink2).

---

## Installation

```bash
mamba create -n AlphaPulldownSnakemake -c conda-forge -c bioconda python=3.12 \
  snakemake snakemake-executor-plugin-slurm snakedeploy pulp click coincbc
mamba activate AlphaPulldownSnakemake
```

### Singularity

Ensure Singularity is installed and available on your system.
See the [official installation guide](https://docs.sylabs.io/guides/latest/user-guide/quick_start.html#quick-installation-steps).

---

## Deploy workflow

```bash
snakedeploy deploy-workflow https://github.com/KosinskiLab/AlphaPulldownSnakemake \
  AlphaPulldownSnakemake --tag 2.1.3
cd AlphaPulldownSnakemake
```

---

## Configuration

Adjust `config/config.yaml` for your particular use case:
```yaml
# List of input sample sheets
input_files:
  - config/sample_sheet.csv

# Delimiter used in protein names
protein_delimiter: "+"

# Directory where all output files will be stored
output_directory: /path/to/output/directory

# Path to AlphaFold database containing required (backend) weights and files
# Note prior to version 2.0.4 this was called alphafold_data_directory
databases_directory: /scratch/AlphaFold_DBs/2.3.2
backend_weights_directory : /scratch/AlphaFold_DBs/2.3.2

# Directories containing precomputed features
feature_directory:
  - "/path/to/directory/with/features1"
  - "/path/to/directory/with/features2"

# If True, only generate features without running structure prediction
only_generate_features: False

# Whether to enable job clustering
cluster_jobs: False

# Bin size for clustering
clustering_bin_size: 150

# Arguments for feature generation
create_feature_arguments:
  --save_msa_files: False  # Save multiple sequence alignment (MSA) files
  --use_precomputed_msas: True  # Use precomputed MSA files if available
  --max_template_date: 2050-01-01  # Set maximum template date to include all templates
  --compress_features: False  # Do not compress generated features
  --data_pipeline: alphafold2 # Use alphafold2 or alphafold3 data pipeline for generating fatures

# Arguments for structure inference
structure_inference_arguments:
  --num_predictions_per_model: 5  # Number of predictions per model
  --num_cycle: 24  # Number of recycles during structure prediction
  --fold_backend: alphafold # Use alphafold2 for predictions

# Arguments for structure analysis
analyze_structure_arguments:
  --cutoff: 100.0  # Cutoff for structure analysis

# Arguments for report generation
generate_report_arguments:
  --cutoff: 100.0  # Cutoff for structure report generation

# Memory allocation settings for feature creation and structure inference
feature_create_ram_bytes: 64000
feature_create_ram_scaling: 1.1
structure_inference_ram_bytes: 32000

# Number of threads for AlphaFold inference
alphafold_inference_threads: 8

# SLURM parameters for inference execution
alphafold_inference: >
  gres=gpu:1 partition=gpu-el8
  qos=normal constraint=gpu=3090

# Specify the backend by changing the prediction container
# (you can also use local singularity .sif files)
# - "docker://kosinskilab/fold:2.1.2" for AlphaFold2
# - "docker://kosinskilab/alphafold3:2.1.2" for AlphaFold3
# - "docker://kosinskilab/alphalink:2.1.2" for AlphaLink2
# - "/path/to/my/container.sif"
prediction_container: "docker://kosinskilab/fold:2.1.2"

# Container for structure analysis
analysis_container: "docker://kosinskilab/fold_analysis:2.1.2"
```

### input_files
This variable holds the path to your sample sheet, where each line corresponds to a folding job. For this pipeline we use the following format specification:

```
protein:N:start-stop[+protein:N:start-stop]*
```

where protein is a path to a file with '.fasta' extension or uniprot ID, N is the number of monomers for this particular protein and start and stop are the residues that should be predicted. However, only protein is required, N, start and stop can be omitted. Hence the following folding jobs for the protein example containing residues 1-50 are equivalent:

```
example:2
example+example
example:2:1-50
example:1-50+example:1-50
example:1:1-50+example:1:1-50
```

This format similarly extends for the folding of heteromers:

```
example1+example2
```

Assuming you have two sample sheets config/sample_sheet1.csv and config/sample_sheet2.csv. The following would be equivalent to computing all versus all in sample_sheet1.csv:

```
input_files :
  - config/sample_sheet1.csv
  - config/sample_sheet1.csv
```

while the snippet below would be equivalent to computing the pulldown between sample_sheet1.csv and sample_sheet2.csv

```
input_files :
  - config/sample_sheet1.csv
  - config/sample_sheet2.csv
```

This format can be extended to as many files as you would like, but keep in mind the number of folds will increase dramatically.

```
input_files :
  - config/sample_sheet1.csv
  - config/sample_sheet2.csv
  - ...
```

### alphafold_data_directory
This is the path to your alphafold database.

### output_directory
Snakemake will write the pipeline output to this directory. If it does not exist, it will be created.

### save_msa, use_precomputed_msa, predictions_per_model, number_of_recycles, report_cutoff
Command line arguments that were previously passed to AlphaPulldown's run_multimer_jobs.py and create_notebook.py (report_cutoff).

### alphafold_inference_threads, alphafold_inference
Slurm specific parameters that do not need to be modified by non-expert users.

### only_generate_features
If set to True, stops after generating features and does not perform structure prediction and reporting.

---

## Run

```bash
snakemake --executor slurm --use-singularity --rerun-incomplete --rerun-triggers mtime --latency-wait 600 --keep-going \
  --singularity-args "--bind /scratch:/scratch --bind /my/disk:/my/disk --nv" \
  --jobs 10 \
  --restart-times 5 \
  -n
```

Remove `-n` to actually execute.
Adjust `--jobs` and `--latency-wait` depending on your cluster/filesystem.

---
Here's a breakdown of what each argument does:
- `--executor slurm`: Use [Snakemake executor plugin](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html) for submitting jobs to a SLURM cluster.

- `--use-singularity`: Enables the use of Singularity containers. This allows for reproducibility and isolation of the pipeline environment.

- `--singularity-args`: Specifies arguments passed directly to Singularity. In the provided example:
  - `--bind /scratch:/scratch` and `--bind /my/disk:/my/disk`: These are bind mount points. They make directories from your host system accessible within the Singularity container. `--nv` ensures the container can make use of the hosts GPUs.

- `--rerun-triggers mtime`: Reruns a job if a specific file (trigger) has been modified more recently than the job's output. Here, `mtime` checks for file modification time.

- `--jobs 10`: Allows up to 10 jobs to be submitted to the cluster simultaneously. For the averaged sized protein complexes on the EMBL cluster (within not too busy periods) you can use up to 400 jobs simultaneously.

- `--restart-times 5`: Specifies that jobs can be automatically restarted up to 5 times if they fail.

- `--rerun-incomplete`: Forces the rerun of any jobs that were left incomplete in previous Snakemake runs.

- `--latency-wait 600`: Waits for 600 seconds after a step finishes to check for the existence of expected output files. This can be useful in file-systems with high latencies.

- `-n`: Dry-run flag. This makes Snakemake display the commands it would run without actually executing them. It's useful for testing. To run the pipeline for real, simply remove this flag.

Executing the command above will submit the following jobs to the cluster:

![Snakemake rulegraph](static/dag.png)

## Tips

* Use `--data_pipeline: alphafold3` and switch `prediction_container` for AF3 inputs/outputs.
* Set `only_generate_features: True` to stop after feature generation.
* Bind extra paths in `--singularity-args` if your data lives outside the default mount points.
