# Snakemake adapter for batched local MMseqs2-GPU features

## Goal

Submit missing AlphaFold 3 protein features through bounded GPU MSA shards and
parallel CPU finalization while leaving native feature generation unchanged
unless local MMseqs2-GPU is explicitly enabled.

## Confirmed public seam

The workflow has two command-line seams: a JAX-free `create_batch_msas.py` GPU
stage that writes durable per-protein MSA bundles plus an atomic shard summary,
and a `finalize_batch_features.py` CPU stage that consumes those bundles and runs
native AF3 template/final feature processing.

## Required behavior

1. The mode is opt-in and valid only for the AlphaFold 3 data pipeline.
2. The prediction image supplies a pinned GPU-capable MMseqs2 executable by default. GPU-compatible, padded UniRef90, MGnify, small-BFD, and paired UniProt paths and database identifiers are explicit configuration; no database discovery occurs.
3. Deterministic shards are bounded by query count and residues. Each requests
   one GPU and invokes exactly one core MSA batch, so databases load once per GPU
   allocation.
4. Existing precomputed feature symlinks remain outside generated batches.
5. The normal one-FASTA `create_features` rule remains unchanged when the mode is disabled.
6. The adapter supplies output compression, local temporary storage, both shard
   limits, and immutable MSA/template database identities to AlphaPulldown.
7. Per-protein MSA bundles are durable side effects, not outputs of the fallible
   shard rule. Only the atomic completion summary is declared, so failed retries
   preserve and reuse validated successes. Final AF3 artifacts are ordinary
   per-protein outputs. The summary records bundle sizes, nanosecond mtimes, and
   SHA-256 digests. Matching stat metadata is the scalable validation fast path;
   changed metadata triggers a streaming digest check. A missing or mismatched
   bundle selects a fresh repair-summary target and reschedules that shard
   without parsing the large bundle JSON. Finalization removes only a bundle
   that fails its semantic validation, making the following DAG self-healing.
8. Only external database, cache, and temporary directories are bound; the bundled binary and other image content remain visible.
9. Search and final caches live in provenance-keyed namespaces so changed
   database/template identities schedule new work under mtime rerun semantics.

## Acceptance criteria

- Deterministic packing honors both limits and keeps an oversized sequence alone.
- Local-MMseqs mode adds one GPU per shard; CPU finalization holds no GPU.
- GPU host RAM is based on database footprint plus one chunk, and runtime scales
  with the shard's sequences and residues.
- A failed shard leaves completed MSA bundles intact and its retry reuses them.
- Disabled mode preserves the current DAG and command.
- AF2 plus local-MMseqs mode fails during workflow validation with a useful message.
