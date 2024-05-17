""" Dedicated rules to cluster jobs and get the number of 
    rows of MSA and the number of total residues to pad 

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Dingquan Yu <dingquan.yu@embl-hamburg.de>
"""
from os.path import join
configfile: "config/config.yaml"


rule all:
    input: join(config["output_directory"], "grouped_jobs")

rule group_jobs_is_done:
    input: "test_data/test_jobs.txt"
    output:
        directory(join(config["output_directory"], "grouped_jobs"))
    params:
        feature_pickle_dir = "test_data/",
        mode = "custom",
        output_dir = config["output_directory"],
    conda:
        "alphapulldown"
    resources:
        avg_mem = lambda wildcards, attempt: 600 * attempt,
        mem_mb = lambda wildcards, attempt: 800 * attempt,
        walltime = lambda wildcards, attempt: 10 * attempt,
        attempt = lambda wildcards, attempt: attempt,
        slurm = "qos=high",
    conda: 
        "alphapulldown"
    
    shell:"""
    source activate alphapulldown
    python src/split_jobs_into_clusters.py --protein_lists={input} --mode={params.mode} \
        --output_dir={params.output_dir} --features_directory={params.feature_pickle_dir}
    """