# Domain glossary

- **Fold**: one structure-prediction request; its chains compose one complex.
- **Inference batch**: the existing size-binned set of independent folds sharing
  one Slurm allocation and one runner configuration. AlphaFold2 monomer and
  multimer folds therefore belong to separate batches.
- **Resident inference**: one batch command keeps AlphaPulldown model runners
  initialized while it executes every fold in an inference batch.
- **Batch manifest**: the JSONL handoff from the workflow to AlphaPulldown; one
  record represents one fold and its output directory.
- **Batch identity**: a singleton's historical fold name, or for a resident batch,
  a bounded `batch-<sha256>` filesystem component derived from its complete ordered
  membership. Composition changes therefore select new workflow artifacts under
  mtime-only rerun triggers.
- **Batch sentinel**: the completion marker named by the batch identity. A resident
  sentinel may live in a synthetic prediction directory; singleton paths are unchanged.
