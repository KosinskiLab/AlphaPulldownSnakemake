#!/bin/bash

snakemake \
  --use-conda \
  --jobs 200 \
  --restart-times 5 \
  --rerun-incomplete \
  --rerun-triggers mtime \
  --latency-wait 30
