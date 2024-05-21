""" Snakemake pipeline for automated structure prediction using various backends.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Dingquan Yu <dingquan.yu@embl-hamburg.de>
"""
## Tried to import InputParser but failed. No module called "workflow"
# from workflow.Snakefile import InputParser
from os.path import join, abspath
configfile: "config/config.yaml"
config["output_directory"] = abspath(config["output_directory"])

rule group_jobs:
    input: 
        join(config["output_directory"], "features"),
    output:
        join(config["output_directory"], "job_groups","job_clusters.txt"),
    params:
        feature_pickle_dir = join(config["output_directory"], "features"),
        output_dir = join(config["output_directory"], "job_groups"),
        all_folds = "A0A075B6L2:2 A0A075B6L2+P0DPR3 P0DPR3+P0DPR3+A0A075B6L2 A0A075B6L2:6"
    resources:
        avg_mem = lambda wildcards, attempt: 600 * attempt,
        mem_mb = lambda wildcards, attempt: 800 * attempt,
        walltime = lambda wildcards, attempt: 10 * attempt,
        attempt = lambda wildcards, attempt: attempt,
        slurm = "qos=high",
    conda: 
        "alphapulldown"
    
    shell:"""
    python src/split_jobs_into_clusters.py  --all_folds {params.all_folds} \
        --output_dir {params.output_dir} --features_directory {params.feature_pickle_dir}
    cd {params.output_dir}
    if [ -f "job_clusters.txt" ]; then
        rm "job_clusters.txt"
    fi
    ls job_cluster*.txt | \
    awk -F '[_/.]' '{{print $3","$4","$5}}' > {output}
    """