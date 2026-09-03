# Snakemake adapter for batched local MMseqs2-GPU features

## Goal

Submit missing AlphaFold 3 protein features through one GPU-backed batch operation while leaving native feature generation unchanged unless local MMseqs2-GPU is explicitly enabled.

## Confirmed public seam

The workflow adapter supplies the ordered set of required protein FASTAs to AlphaPulldown's `FeatureBatch` command-line seam. Deduplication and chunking constrained by sequence count and total residues remain behind that deep interface.

## Required behavior

1. The mode is opt-in and valid only for the AlphaFold 3 data pipeline.
2. UniRef90, MGnify, small-BFD, and paired UniProt paths and database identifiers are explicit configuration; no database discovery occurs.
3. The batch operation requests one GPU, configured feature threads, and memory sized for its peak query chunk.
4. Existing precomputed feature symlinks remain outside generated batches.
5. The normal one-FASTA `create_features` rule remains unchanged when the mode is disabled.
6. The adapter supplies output compression, local temporary storage, and both chunk limits to AlphaPulldown.
7. AlphaPulldown decides cache validity and skips only matching artifacts; every required named artifact remains an observable Snakemake output.

## Acceptance criteria

- Deterministic packing honors both limits and keeps an oversized sequence alone.
- Local-MMseqs mode adds exactly one GPU resource and the required command-line settings.
- Disabled mode preserves the current DAG and command.
- AF2 plus local-MMseqs mode fails during workflow validation with a useful message.
