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
  AlphaPulldownSnakemake --tag 2.1.2
cd AlphaPulldownSnakemake
```

---

## Configuration

Edit `config/config.yaml` (minimal example):

```yaml
input_files:
  - config/sample_sheet.csv

output_directory: /path/to/output
databases_directory: /scratch/AlphaFold_DBs/2.3.2

create_feature_arguments:
  --use_precomputed_msas: True
  --max_template_date: 2050-01-01
  --compress_features: False
  --data_pipeline: alphafold2   # or alphafold3

structure_inference_arguments:
  --num_predictions_per_model: 5
  --num_cycle: 24
  --fold_backend: alphafold     # or alphafold3 or alphalink

alphafold_inference_threads: 8

# Choose backend container (or a local .sif path)
prediction_container: "docker://kosinskilab/fold:2.1.2"        # AF2
# prediction_container: "docker://kosinskilab/alphafold3:2.1.2" # AF3
# prediction_container: "docker://kosinskilab/alphalink:2.1.2" # AL2
analysis_container: "docker://kosinskilab/fold_analysis:2.1.2"

only_generate_features: False
```

---

## Input format

This variable holds the path to your sample sheet, where each line corresponds to a folding job. For this pipeline we use the following format specification:

```
protein:N:start-stop[_protein:N:start-stop]*
```

where protein is a path to a file with '.fasta' extension or uniprot ID, N is the number of monomers for this particular protein and start and stop are the residues that should be predicted. However, only protein is required, N, start and stop can be omitted. Hence the following folding jobs for the protein example containing residues 1-50 are equivalent:

```
example:2
example_example
example:2:1-50
example:1-50_example:1-50
example:1:1-50_example:1:1-50
```

This format similarly extends for the folding of heteromers:

```
example1_example2
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

- `--use-singularity`: Enables the use of Singularity containers. This allows for reproducibility and isolation of the pipeline environment.

- `--singularity-args`: Specifies arguments passed directly to Singularity. In the provided example:
  - `--bind /scratch:/scratch` and `--bind /my/disk:/my/disk`: These are bind mount points. They make directories from your host system accessible within the Singularity container. `--nv` ensures the container can make use of the hosts GPUs.

- `--rerun-triggers mtime`: Reruns a job if a specific file (trigger) has been modified more recently than the job's output. Here, `mtime` checks for file modification time.

- `--jobs 500`: Allows up to 500 jobs to be submitted to the cluster simultaneously.

- `--restart-times 10`: Specifies that jobs can be automatically restarted up to 10 times if they fail.

- `--rerun-incomplete`: Forces the rerun of any jobs that were left incomplete in previous Snakemake runs.

- `--latency-wait 30`: Waits for 30 seconds after a step finishes to check for the existence of expected output files. This can be useful in file-systems with high latencies.

- `-n`: Dry-run flag. This makes Snakemake display the commands it would run without actually executing them. It's useful for testing. To run the pipeline for real, simply remove this flag.

Executing the command above will submit the following jobs to the cluster:

![Snakemake rulegraph](static/dag.png)

## Tips

* Use `--data_pipeline: alphafold3` and switch `prediction_container` for AF3 inputs/outputs.
* Set `only_generate_features: True` to stop after feature generation.
* Bind extra paths in `--singularity-args` if your data lives outside the default mount points.
