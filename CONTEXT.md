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
- **Feature request**: one named sequence requiring a feature artifact for the configured
  backend.
- **MSA shard**: a deterministic, count- and residue-bounded group submitted as
  one JAX-free MMseqs2 batch, on GPU or CPU. Unknown-length requests run alone.
- **Shard registry**: the append-only record, beside the MSA cache, of every MSA shard
  ever planned in that cache namespace. A protein keeps the shard it was first planned
  into, and every registered shard's job id keeps resolving, so every parse of a run -
  the head node's and each SLURM job's - agrees on which job computes which protein.
- **MSA shard schedule**: which shard job computes each requested protein this parse,
  derived from the shard registry and the Shard completions. A registered shard runs
  only when all its proteins are still requested; a partly requested one that already
  completed keeps serving its remaining proteins.
- **MSA bundle**: a durable per-protein cache artifact containing paired and
  unpaired alignments plus MMseqs2/search/database provenance.
- **Shard completion**: an atomic summary written only after every request in an
  MSA shard succeeds. The MSA bundles are not declared outputs, so a failed shard
  retry retains and validates prior successes.
- **Feature finalization**: a per-protein CPU job that consumes an MSA bundle, runs
  the backend's template search, and writes its standard feature artifact: AF3's
  native template processing and feature JSON, or AF2's hmmsearch/hhsearch and a
  MonomericObject pickle.
- **Database identifier**: the configured immutable identity of one MMseqs2 database build.
- **Feature artifact**: the standard per-protein features structure inference consumes:
  an AlphaFold 3 JSON, or an AlphaFold 2 MonomericObject pickle.
- **Cache hit**: an artifact that AlphaPulldown validates against the sequence,
  MMseqs2 executable version, output-affecting settings, database identifiers,
  template cutoff, and template database identifiers as appropriate.

