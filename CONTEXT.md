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
- **Feature request**: one named protein sequence requiring an AlphaFold 3 feature artifact.
- **MSA shard**: a deterministic, count- and residue-bounded group submitted as
  one JAX-free GPU MMseqs2 batch. Unknown-length requests run alone.
- **MSA bundle**: a durable per-protein cache artifact containing paired and
  unpaired alignments plus MMseqs2/search/database provenance.
- **Shard completion**: an atomic summary written only after every request in an
  MSA shard succeeds. The MSA bundles are not declared outputs, so a failed shard
  retry retains and validates prior successes.
- **Feature finalization**: a per-protein CPU job that consumes an MSA bundle,
  runs native AF3 template processing, and writes the standard AF3 feature JSON.
- **Database identifier**: the configured immutable identity of one MMseqs2 database build.
- **Feature artifact**: the standard per-protein AlphaFold 3 JSON consumed by structure inference.
- **Cache hit**: an artifact that AlphaPulldown validates against the sequence,
  MMseqs2 executable version, output-affecting settings, database identifiers,
  template cutoff, and template database identifiers as appropriate.

