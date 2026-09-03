# Domain glossary

- **Fold**: one structure-prediction request; its chains compose one complex.
- **Inference batch**: the existing size-binned set of independent folds sharing
  one Slurm allocation and one runner configuration. AlphaFold2 monomer and
  multimer folds therefore belong to separate batches.
- **Resident inference**: one batch command keeps AlphaPulldown model runners
  initialized while it executes every fold in an inference batch.
- **Batch manifest**: the JSONL handoff from the workflow to AlphaPulldown; one
  record represents one fold and its output directory.
- **Batch sentinel**: the existing completion marker for an inference batch.
