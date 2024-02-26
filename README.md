# Alphaabriss

A snakemake pipeline for automated structure prediction using various backends.

## Installation

1. **Snakemake**: An integral workflow management system.

   ```bash
   pip install snakemake
   ```
2. **Cluster Setup**

   In order to allow snakemake to interface with a compute cluster, we are going to use the [Snakemake-Profile for SLURM](https://github.com/Snakemake-Profiles/slurm). If you are not working on a SLURM cluster you can find profiles for different architectures [here](https://github.com/Snakemake-Profiles/slurm). The following will create a profile that can be used with snakemake and prompt you for some additional information.

   ```bash
   pip install cookiecutter
   git clone https://github.com/Snakemake-Profiles/slurm.git
   profile_dir="${HOME}/.config/snakemake"
   mkdir -p "$profile_dir"
   template="gh:Snakemake-Profiles/slurm"
   cookiecutter --output-dir "$profile_dir" "$template"
   ```

3. **Singularity**: We make use of singularity containers in this pipeline. If you have not installed singularity, check the [official Singularity guide](https://sylabs.io/guides/latest/user-guide/quick_start.html#quick-installation-steps). If you are working on the EMBL cluster singularity is already installed and you can skip this step.


## Configuration

Adjust `config/config.yaml` for your particular use case.

### input_file
This file contains your sample sheet where each line corresponds to a folding job. For this pipeline we use the following format specification:

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

## Execution

```bash
snakemake \
  --use-singularity \
  --singularity-args "-B /scratch:/scratch \
    -B /g/kosinski:/g/kosinski \
    --nv " \
  --jobs 200 \
  --restart-times 5 \
  --profile slurm_noSidecar \
  --rerun-incomplete \
  --rerun-triggers mtime \
  --latency-wait 30 \
  -n

```

Here's a breakdown of what each argument does:

- `--use-singularity`: Enables the use of Singularity containers. This allows for reproducibility and isolation of the pipeline environment.

- `--singularity-args`: Specifies arguments passed directly to Singularity. In the provided example:
  - `-B /scratch:/scratch` and `-B /g/kosinski:/g/kosinski`: These are bind mount points. They make directories from your host system accessible within the Singularity container. `--nv` ensures the container can make use of the hosts GPUs.

- `--profile name_of_your_profile`: Specifies the Snakemake profile to use (e.g., the SLURM profile you set up for cluster execution).

- `--rerun-triggers mtime`: Reruns a job if a specific file (trigger) has been modified more recently than the job's output. Here, `mtime` checks for file modification time.

- `--jobs 500`: Allows up to 500 jobs to be submitted to the cluster simultaneously.

- `--restart-times 10`: Specifies that jobs can be automatically restarted up to 10 times if they fail.

- `--rerun-incomplete`: Forces the rerun of any jobs that were left incomplete in previous Snakemake runs.

- `--latency-wait 30`: Waits for 30 seconds after a step finishes to check for the existence of expected output files. This can be useful in file-systems with high latencies.

- `-n`: Dry-run flag. This makes Snakemake display the commands it would run without actually executing them. It's useful for testing. To run the pipeline for real, simply remove this flag.

Executing the command above will perform submit the following jobs to the  cluster:

![Snakemake rulegraph](static/dag.png)
