from os.path import abspath
from os import makedirs
include: "group_jobs.smk"
# include: "structure_predict_with_padding.smk"

configfile: "config/config.yaml"
config["output_directory"] = abspath(config["output_directory"])
makedirs(config["output_directory"], exist_ok = True)
base_inference_ram = config.get("structure_inference_ram_bytes", 32000)

rule all:
    input: 
        join(config["output_directory"], "job_groups","job_clusters.txt"),
        "aggregate.txt"


checkpoint structure_inference_with_padding:
    output:
        directory(join(config["output_directory"], "job_groups", "reports"))
    params:
        data_directory=config["alphafold_data_directory"],
        predictions_per_model=config["predictions_per_model"],
        n_recycles=lambda wildcards: min(3, config["number_of_recycles"]),
        feature_directory=join(config["output_directory"], "features"),
        output_directory=join(config["output_directory"], "predictions"),
        job_group_directory=join(config["output_directory"], "job_groups")
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
        
        DIRECTORY={params.job_group_directory}
        for FILE in "$DIRECTORY"/job_cluster_*_*.txt; do
            FILENAME=$(basename "$FILE")
            if [[ $FILENAME =~ job_cluster_([0-9]+)_([0-9]+)_([0-9]+).txt ]]; then
                CLUSTER_INDEX=${{BASH_REMATCH[1]}}
                NUM_DESIRED_RES=${{BASH_REMATCH[2]}}
                NUM_DESIRED_MSA=${{BASH_REMATCH[3]}}
                run_multimer_jobs.py \
                    --mode custom \
                    --output_path={params.output_directory} \
                    --num_cycle={params.n_recycles} \
                    --num_predictions_per_model={params.predictions_per_model} \
                    --data_dir={params.data_directory} --monomer_objects_dir={params.feature_directory} \
                    --protein_list=$FILENAME \
                    --desired_num_res=$NUM_DESIRED_RES \
                    --desired_num_msa=$NUM_DESIRED_MSA

                echo "Completed" > {output}/$FILENAME
            fi
        done
        """


def dynamic_update(wildcards):  
    checkpoint_output = checkpoints.structure_inference_with_padding.get(**wildcards).output[0]
    CLUSTER_INDEX, NUM_DESIRED_RES, NUM_DESIRED_MSA = glob_wildcards(join(checkpoint_output, "job_cluster_{cluster_index}_{num_desired_res}_{num_desired_msa}.txt"))
    return expand(
        join(checkpoint_output, "job_cluster_{cluster_index}_{num_desired_res}_{num_desired_msa}.txt"),
        cluster_index=CLUSTER_INDEX,
        num_desired_res=NUM_DESIRED_RES,
        num_desired_msa=NUM_DESIRED_MSA
    )

rule aggregate:
    input:
        dynamic_update
    output:
        "aggregate.txt"

    shell:"""
    cat {input} > {output}
    """