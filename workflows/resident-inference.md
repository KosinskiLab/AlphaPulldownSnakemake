# Resident inference workflow integration

The AlphaPulldown `PredictionBatch` interface and JSONL contract are the source
of truth. This workflow maps the existing size-binned Snakemake batch to it.

For `batch_size > 1`, write one manifest line per member of `BATCH_FOLDS` in its
existing order and invoke `run_structure_prediction_batch.py` once in the
`structure_inference` Slurm allocation. Preserve token sorting,
`batch_max_tokens`, largest-fold memory and GPU-tier sizing, count-scaled runtime,
all backend flags, per-fold output directories, retry behavior, and the existing
batch sentinel. AlphaFold2 monomer and multimer folds require different model
runners and are therefore size-binned separately; AlphaFold3 grouping is unchanged.

For `batch_size <= 1`, invoke `run_structure_prediction.py` exactly as before so
the default remains compatible with older prediction containers.

Acceptance is observable at the generated manifest and command seam: two folds
produce two JSONL records and one batch command, while one fold produces no
manifest dependency and the legacy command. Paths written into the manifest are
interpreted relative to its parent directory.
