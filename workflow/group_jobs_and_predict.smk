""" Snakemake pipeline for automated structure prediction using various backends.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Dingquan Yu <dingquan.yu@embl-hamburg.de>
"""

from os.path import abspath, join
from os import makedirs, listdir
# include: "group_jobs.smk"
import re

configfile: "config/config.yaml"
config["output_directory"] = abspath(config["output_directory"])
makedirs(config["output_directory"], exist_ok = True)
base_inference_ram = config.get("structure_inference_ram_bytes", 32000)

required_job_clusters = [i for i in listdir(join(config["output_directory"], 
                                            "job_groups")) if i.startswith("job_cluster_") and i.endswith(".txt")]
required_reports = []
for cluster in required_job_clusters:
    pattern = r'job_cluster_(\d+)_(\d+)_(\d+)\.txt'
    matched = re.match(pattern, cluster)
    if matched:
        cluster_index = matched.group(1)
        num_desired_res = matched.group(2)
        num_desired_msa = matched.group(3)
        required_reports.append(join(config["output_directory"], 
                                            "job_groups", "reports",f"cluster_{cluster_index}_{num_desired_res}_{num_desired_msa}.txt")) 
rule all:
    input: 
        [*required_reports]


rule structure_inference_with_padding:
    input:
        join(config["output_directory"], "job_groups", 
        "job_cluster_{cluster_index}_{num_desired_res}_{num_desired_msa}.txt")
    output:
        join(config["output_directory"], "job_groups", "reports",
        "cluster_{cluster_index}_{num_desired_res}_{num_desired_msa}.txt")
    params:
        data_directory=config["alphafold_data_directory"],
        predictions_per_model=config["predictions_per_model"],
        n_recycles=lambda wildcards: min(3, config["number_of_recycles"]),
        feature_directory=join(config["output_directory"], "features"),
        output_directory=join(config["output_directory"], "predictions"),
        num_desird_res=lambda wildcards: wildcards.num_desired_res,
        num_desired_msa=lambda wildcards: wildcards.num_desired_msa,
    resources:
        mem_mb=lambda wildcards, attempt: base_inference_ram * (1.1 ** attempt),
        walltime=lambda wildcards, attempt: 1440 * attempt,
        attempt=lambda wildcards, attempt: attempt,
        slurm=config.get("alphafold_inference", "")
    threads:
        config["alphafold_inference_threads"]
    container:
        "docker://kosinskilab/fold:latest"
    shell:"""
        #MAXRAM=$(bc <<< "$(ulimit -m) / 1024.0")
        #GPUMEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tail -1)
        #export XLA_PYTHON_CLIENT_MEM_FRACTION=$(echo "scale=3; $MAXRAM / $GPUMEM" | bc)
        #export TF_FORCE_UNIFIED_MEMORY='1'
        
        run_multimer_jobs.py \
            --mode custom \
            --output_path={params.output_directory} \
            --num_cycle={params.n_recycles} \
            --num_predictions_per_model={params.predictions_per_model} \
            --data_dir={params.data_directory} --monomer_objects_dir={params.feature_directory} \
            --protein_lists={input} \
            --desired_num_res={params.num_desird_res} \
            --desired_num_msa={params.num_desired_msa}

        echo "Completed" > {output}
        """