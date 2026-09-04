# MMseqs2 memory scaling: target database vs query load

Research note backing the memory model for the MMseqs2-GPU MSA stage. Question:
is a per-residue (or per-query) term justified, or is peak memory essentially a
function of the target database alone?

**Short answer: peak host memory is a function of the target database, the
thread count, and the k-mer index — not of the query set.** Every query-side
buffer on the search path is allocated once, outside the query loop, and sized
by the `--max-seq-len` *parameter* rather than by any actual query. A
per-residue term in a memory model is not supported by the sources. A
per-residue term in a *runtime* model is.

All source citations are pinned to release `18-8cc5c`, commit
[`8cc5ce367b5638c4306c2d7cfc652dd099a4643f`](https://github.com/soedinglab/MMseqs2/tree/8cc5ce367b5638c4306c2d7cfc652dd099a4643f),
which is the release in use here.

---

## 1. The prefilter memory formula, and which terms are target- vs query-derived

### The documented formula

The MMseqs2 User Guide, section *Optimizing sensitivity and consumption of
resources → Memory consumption*:

> For maximum efficiency of the prefiltering, the entire database should be held
> in RAM. The major part of memory is required for the k-mer index table of the
> database. For a database containing `N` sequences with an average length `L`,
> the memory consumption of the index lists is `(N * L * 7) byte`. Note that the
> memory consumption grows linearly with the size of the sequence database. In
> addition, the index table stores the pointer array and two auxiliary arrays
> with the memory consumption of `a^k*8` byte […]
>
> ```
> M = (7 * N * L + 8 a^k) byte
> ```

Source: [MMseqs2 wiki, `Home.md`](https://github.com/soedinglab/MMseqs2/wiki#memory-consumption)
(headings *Prefiltering → Memory consumption*). Note `N * L` is just the total
residue count of the **target** database. Both terms are target-only.

### The implemented formula

`Prefiltering::estimateMemoryConsumption` is the function that actually decides
database splitting:

[`src/prefiltering/Prefiltering.cpp#L1067-L1098`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/prefiltering/Prefiltering.cpp#L1067-L1098)

```cpp
size_t Prefiltering::estimateMemoryConsumption(int split, size_t dbSize, size_t resSize,
                                               size_t maxResListLen,
                                               int alphabetSize, int kmerSize, unsigned int querySeqType,
                                               int threads) {
    // for each residue in the database we need 7 byte
    size_t dbSizeSplit = (dbSize) / split;
    size_t residueSize = (resSize / split * 7);
    // 21^7 * pointer size is needed for the index
    size_t indexTableSize = static_cast<size_t>(pow(alphabetSize, kmerSize)) * sizeof(size_t);
    // memory needed for the threads
    size_t threadSize = threads * (
            (dbSizeSplit * 2 * sizeof(IndexEntryLocal))
            + (dbSizeSplit * 1.5 * sizeof(CounterResult))
            + (maxResListLen * sizeof(hit_t))
            + (dbSizeSplit * 2 * sizeof(CounterResult) * 2)
    );
    size_t dbReaderSize = dbSize * (sizeof(DBReader<unsigned int>::Index) + sizeof(unsigned int));
    size_t extendedMatrix = 0;
    if(Parameters::isEqualDbtype(querySeqType, Parameters::DBTYPE_AMINO_ACIDS)){
        extendedMatrix = sizeof(std::pair<short, unsigned int>) * static_cast<size_t>(pow(pow(alphabetSize, 3), 2));
        extendedMatrix += sizeof(std::pair<short, unsigned int>) * pow(pow(alphabetSize, 2), 2);
    }
    size_t background = dbSize * 22;
    return residueSize + indexTableSize + threadSize + background + extendedMatrix + dbReaderSize;
}
```

**Term-by-term attribution.** `dbSize` is `tdbr.getSize()` (number of *target*
sequences) and `resSize` is `tdbr.getAminoAcidDBSize()` (total *target*
residues) at both call sites
([L276-L278](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/prefiltering/Prefiltering.cpp#L276-L278),
[L367-L368](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/prefiltering/Prefiltering.cpp#L367-L368)):

| Term | Scales with | Query-dependent? |
| --- | --- | --- |
| `residueSize` = `7 * R / split` | target residues `R` | no |
| `indexTableSize` = `a^k * 8` | alphabet & k-mer size only | no |
| `threadSize` ≈ `T * (N/split) * 50.5 B` | threads × target seqs `N` | no |
| `dbReaderSize`, `background` | target seqs `N` | no |
| `extendedMatrix` = `8*(a^3)^2 + 8*(a^2)^2` (~0.7 GB for `a`=21) | constant | no |
| `maxResListLen * sizeof(hit_t)` per thread | `--max-seqs` | no |

**There is no query term in the formula at all.** The signature does not even
accept the query database size.

The 50.5 B/target-sequence/thread figure is derived from the packed struct
definitions: `IndexEntryLocal` is `__attribute__((__packed__))` `unsigned int` +
`unsigned short` = 6 B
([`IndexTable.h#L25`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/prefiltering/IndexTable.h#L25));
`CounterResult` is packed `unsigned int` + `unsigned short` + `unsigned char` =
7 B
([`CacheFriendlyOperations.h#L46`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/prefiltering/CacheFriendlyOperations.h#L46)).
So `2*6 + 1.5*7 + 2*7*2 = 50.5`. *(The arithmetic is mine; the struct
definitions and the expression are sourced.)*

That per-thread term is a real allocation, not just an accounting estimate.
`QueryMatcher`'s constructor sizes its arrays from the target `dbSize`, and one
`QueryMatcher` exists per thread:

[`src/prefiltering/QueryMatcher.cpp#L40-L46`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/prefiltering/QueryMatcher.cpp#L40-L46)

```cpp
// this array will need 500 MB for 50 Mio. sequences ( dbSize * 2 * 5byte)
this->dbSize = dbSize;
this->foundDiagonalsSize = std::max((size_t)1000000, dbSize);
this->maxDbMatches = std::max((size_t)1000000, dbSize) * 2;
```

**Consequence worth flagging:** CPU prefilter memory scales with
`threads × target_sequences`. On a 64-thread node against a 200 M-sequence
database that term alone is ~646 GB. Thread count is a first-class memory
parameter on the CPU path, and is absent from the wiki's `M` formula.

### Where the query database *does* appear

Exactly one place: capping the number of splits when `--split-mode 1`
(query-split) is selected. It never enters a memory calculation.

[`src/prefiltering/Prefiltering.cpp#L326-L327`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/prefiltering/Prefiltering.cpp#L326-L327)

```cpp
    if (splitMode == Parameters::QUERY_DB_SPLIT) {
        sizeOfDbToSplit = qDbSize;
    }
```

---

## 2. Is query memory negligible? At what load does that change?

Yes, and on the GPU path there is no query load at which it stops being true,
because **queries are processed strictly one at a time**.

The GPU search loop, with every buffer allocated *before* it and reused:

[`src/prefiltering/ungappedprefilter.cpp#L47-L68`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/prefiltering/ungappedprefilter.cpp#L47-L68)
and
[`#L164-L199`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/prefiltering/ungappedprefilter.cpp#L164-L199)

```cpp
Sequence qSeq(par.maxSeqLen, querySeqType, subMat, 0, false, par.compBiasCorrection);
std::vector<Marv::Result> results;
results.reserve(par.maxResListLen);
size_t profileBufferLength = par.maxSeqLen;
profile = (int8_t*)malloc(subMat->alphabetSize * profileBufferLength * sizeof(int8_t));
...
for (size_t id = 0; id < qdbr->getSize(); id++) {
    ...
    stats = marv->scan(..., qSeq.L, profile, results.data());
}
```

The design is stated in the User Guide too:

> GPU-accelerated searches parallelize computation within a single query rather
> than across multiple queries (default of MMseqs2 CPU), enabling very fast
> searches even for small protein sets.

Source: [MMseqs2 wiki, *GPU-accelerated search*](https://github.com/soedinglab/MMseqs2/wiki#gpu-accelerated-search).

This is a complete, sourced explanation for the measured flatness at 149.4 GB
across shards of 1, 8, 32 and 128 queries. Adding queries adds loop iterations,
not allocations. **The measurement is exactly what the code predicts.**

Scale at which the query set could matter:

- **GPU path: never, for host RAM.** Cost is `O(1)` in query count. Query load
  buys runtime, not memory.
- **CPU path: also effectively never for MSA-scale shards.** The only
  query-influenced sizing is `--split-mode 1` (query-split), and that *reduces*
  memory. Query sequence data is streamed from the DBReader.
- The real query-count limit is **output size**, not memory: prefilter results
  are `~21 bytes` per hit and MMseqs2 has a separate
  `estimateHDDMemoryConsumption` = `2 * (21 * dbSize * maxResListLen)`
  ([`Prefiltering.cpp#L1100-L1104`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/prefiltering/Prefiltering.cpp#L1100-L1104)).
  That is disk, not RAM.

---

## 3. `--split`, `--split-memory-limit`, `--db-load-mode`, and what RSS means

### `--split` / `--split-memory-limit`

> `--split-mode` — "0: split target db; 1: split query db; 2: auto, depending on main memory"
> `--split-memory-limit` — "Set max memory per split. E.g. 800B, 5K, 10M, 1G. Default (0) to all available system memory"

Source: [`src/commons/Parameters.cpp#L55-L56`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/commons/Parameters.cpp#L55-L56).

The guide adds the calibration caveat:

> The `--split-memory-limit` parameter can give MMseqs2 an upper limit of system
> RAM to use for the large prefiltering data structures. MMseqs2 will still use
> some additional memory for its database structures etc. In total,
> `--split-memory-limit` will be about `80%` of the total memory required.

Source: [MMseqs2 wiki, *Memory consumption*](https://github.com/soedinglab/MMseqs2/wiki#memory-consumption).
So a job that must fit in `X` should pass roughly `0.8X`.

**Trap for SLURM users.** With the default `--split-memory-limit 0`, the limit is
90% of *physical node memory*, discovered via `sysconf(_SC_PHYS_PAGES)` — which
is **not** cgroup- or SLURM-aware:

[`src/commons/Util.cpp#L584-L599`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/commons/Util.cpp#L584-L599)
and [`#L293-L315`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/commons/Util.cpp#L293-L315)

```cpp
memoryLimit = static_cast<size_t>(Util::getTotalSystemMemory() * 0.9);
...
static size_t phys_pages = sysconf(_SC_PHYS_PAGES);
```

On a 755 GB `htc-el8` node with `--mem=64G`, MMseqs2 will size itself for ~680 GB
and decline to split. *(That `sysconf` ignores cgroup limits is standard Linux
behaviour, not something the MMseqs2 sources state — but the call site is
sourced above.) Recommendation: always pass an explicit `--split-memory-limit`
on the CPU path.*

### `--db-load-mode` and whether RSS is reclaimable

> "Database preload mode 0: auto, 1: fread, 2: mmap, 3: mmap+touch"

Source: [`src/commons/Parameters.cpp#L67`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/commons/Parameters.cpp#L67).

This distinction is real and it changes the *kind* of memory, which is the crux
of the question about `/usr/bin/time %M`.

**mmap path (modes 0/2/3):** `MAP_PRIVATE`, `PROT_READ` against the DB file —
file-backed, clean pages.

[`src/commons/DBReader.cpp#L429-L451`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/commons/DBReader.cpp#L429-L451)

```cpp
if ((dataMode & USE_FREAD) == 0) {
    ...
    ret = static_cast<char*>(mmap(NULL, *dataSize, mode, MAP_PRIVATE, fd, 0));
} else {
    ret = static_cast<char*>(malloc(*dataSize));
    Util::checkAllocation(ret, "Not enough system memory to read in the whole data file.");
    size_t result = fread(ret, 1, *dataSize, file);
}
```

**fread path (mode 1):** `malloc` + `fread` — anonymous memory. Non-reclaimable,
genuinely reserved.

"Touching" is a read sweep plus `POSIX_MADV_WILLNEED`; it faults pages in but
does **not** copy them out of the page cache:

[`src/commons/Util.cpp#L334-L343`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/commons/Util.cpp#L334-L343)

```cpp
char Util::touchMemory(const char *memory, size_t size) {
    if (size > 0 && posix_madvise ((void*)memory, size, POSIX_MADV_WILLNEED) != 0){ ... }
    if(size > Util::getTotalSystemMemory()){
        Debug(Debug::WARNING) << "Can not touch " << size << " into main memory\n";
        return 0;
    }
```

and the guide's own description:

> `--db-load-mode 2` tells MMseqs2 to `mmap` the database instead of copying the
> whole precomputed index into memory. This saves, for a large database, minutes
> of copying from the storage system into RAM. However, this is less efficient
> for large query sets.

Source: [MMseqs2 wiki, `Home.md` L2894](https://github.com/soedinglab/MMseqs2/wiki).

**So: yes — under modes 0/2/3 the reported peak RSS is dominated by clean,
file-backed page-cache pages that the kernel can evict under pressure, rather
than by memory that must be reserved.** Under mode 1 it is not; that is real
anonymous memory.

Which mode is in force on the GPU path is decided here — and the default (0)
touches:

[`src/prefiltering/ungappedprefilter.cpp#L487-L488`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/prefiltering/ungappedprefilter.cpp#L487-L488)

```cpp
bool touch = (par.preloadMode != Parameters::PRELOAD_MODE_MMAP);
IndexReader tDbrIdx(par.db2, par.threads, IndexReader::SEQUENCES, (touch) ? (IndexReader::PRELOAD_INDEX | IndexReader::PRELOAD_DATA) : 0 );
```

Only `--db-load-mode 2` sets `touch = false`. Modes 0, 1 and 3 all pull the whole
target database into RSS. **The 149.4 GB is a touch sweep of the target database,
and `--db-load-mode 2` should collapse it to whatever the search actually
reads.** That is a cheap, decisive experiment to run.

Note also that on the **CPU** path, `PRELOAD_MODE_AUTO` resolves to a genuine
`fread` at high sensitivity:

[`src/prefiltering/Prefiltering.cpp#L84-L90`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/prefiltering/Prefiltering.cpp#L84-L90)

```cpp
if (preloadMode == Parameters::PRELOAD_MODE_AUTO) {
    if (sensitivity > 6.0) {
        preloadMode = Parameters::PRELOAD_MODE_FREAD;
    } else {
        preloadMode = Parameters::PRELOAD_MODE_MMAP_TOUCH;
    }
}
```

ColabFold-style MSA generation runs at high sensitivity, so a CPU run there
plausibly lands on `fread` — i.e. non-reclaimable. This is part of the answer to
Q6.

---

## 4. GPU mode: VRAM vs host RAM, and the padded database

**What must fit in VRAM: nothing, strictly.** The database is streamed when it
does not fit.

> MMseqs2-GPU efficiently processes databases that exceed GPU memory by streaming
> data directly from host RAM using asynchronous CUDA streams. If the database
> fits into host RAM but exceeds GPU memory, MMseqs2-GPU maintains approximately
> 60% of its peak performance.

Source: [MMseqs2 wiki, *Database larger than GPU memory*](https://github.com/soedinglab/MMseqs2/wiki#database-larger-than-gpu-memory).

The underlying library says the same:

> Depending on the database size and available total GPU memory, the database is
> transferred to the GPU once for all queries, or it is processed in batches
> which requires a transfer for each query. […] For best performance, the
> complete database must fit into `maxGpuMem` times the number of used GPUs.

Source: [`lib/libmarv/Readme.md`, *Memory options*](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/lib/libmarv/Readme.md).

And from the paper:

> Potential GPU memory constraints are mitigated through reduced memory
> footprint, efficient database streaming, partitioning and clustered searches

> [the workflow] partitions the reference database into smaller batches, allowing
> processing to be pipelined via asynchronous CUDA streams

Source: Kallenborn et al., *GPU-accelerated homology search with MMseqs2*,
Nature Methods (2025), [doi:10.1038/s41592-025-02819-8](https://doi.org/10.1038/s41592-025-02819-8),
free full text [PMC12510879](https://pmc.ncbi.nlm.nih.gov/articles/PMC12510879/);
sections *Main* and *Methods → Database streaming*.

**What stays in host RAM: the whole padded database.** `Marv::loadDb` is handed
the host pointer to the mmap'd DB data
([`ungappedprefilter.cpp#L147-L149`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/prefiltering/ungappedprefilter.cpp#L147-L149)).
Streaming *from* host RAM presupposes it is *in* host RAM. This is why the host
figure tracks database size and nothing else.

**Does padding change the host-memory profile?** Only slightly, and it is
bounded. `makepaddedseqdb` pads each sequence up to a 4-residue boundary:

[`src/util/makepaddedseqdb.cpp#L46,L88-L89`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/util/makepaddedseqdb.cpp#L88-L89)

```cpp
const int ALIGN = 4;
...
const size_t sequencepadding = (seq.L % ALIGN == 0) ? 0 : ALIGN - seq.L % ALIGN;
result.append(sequencepadding, static_cast<char>(20));
```

Worst case 3 extra bytes per sequence — under 1% for typical protein lengths.
Padding is for GPU access alignment, not a memory multiplier.

**The 1 byte/residue figure.** This is the key GPU-mode constant:

> The `--index-subset 2` parameter is useful for omitting the large k-mer data
> structures required by the default MMseqs2 k-mer-based search, reducing memory
> usage to approximately 1 byte per residue in the target database instead of 7
> bytes for the full k-mer index.

Source: [MMseqs2 wiki, *GPU-accelerated search*](https://github.com/soedinglab/MMseqs2/wiki#gpu-accelerated-search).
The paper states it as a headline result: "MMseqs2-GPU reduces this memory demand
from ~7 bytes to 1 byte per residue" (Kallenborn et al. 2025, *Main*).

**`gpuserver`.** A persistent process holding the DB resident on the GPU, to
amortise CUDA init:

> We introduced an optional, dedicated GPU server mode […] a persistent
> background process that maintains the GPU context and becomes responsible for
> database caching […] [avoiding] CUDA initialization overhead of approximately
> 300 ms […] during ColabFold MSA search, the ungappedprefilter module is called
> six times.

Source: Kallenborn et al. 2025, *Methods → GPU server*.

Its host-side shared-memory block is sized by parameters, not by the query set
— and this is the single place in the codebase where a per-residue term appears
at all:

[`src/commons/GpuUtil.h#L33-L38`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/commons/GpuUtil.h#L33-L38)

```cpp
static size_t calculateSize(unsigned int maxSeqLen, unsigned int maxResListLen) {
    return sizeof(GPUSharedMemory) +
           sizeof(char) * maxSeqLen +              // Size for query data
           sizeof(Marv::Result) * maxResListLen +  // Size for results data
           sizeof(int8_t) * 21 * maxSeqLen;        // Size for profile data
}
```

22 bytes per residue of `maxSeqLen` — but `maxSeqLen` is the **parameter**
(default 65535), not an actual query, so this is a fixed ~1.4 MB.

Re: the observed ~45 GB of 46 GB L40S — that is consistent with Marv taking all
available VRAM by design. `--maxGpuMem` defaults to "All available gpu memory"
and is explicitly "not a hard limit" (`lib/libmarv/Readme.md`, *Memory options*).
High VRAM occupancy is the allocator working as intended, not a capacity signal.

---

## 5. Does memory grow with individual query LENGTH?

**No — not for any query MMseqs2 will accept.** This is the strongest negative
result in the note, and it is structural rather than empirical.

The query-side buffers are pre-sized to `par.maxSeqLen`, whose default is 2^16:

[`src/commons/Parameters.cpp#L2354-L2355`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/commons/Parameters.cpp#L2354-L2355)

```cpp
maxSeqLen = MAX_SEQ_LEN; // 2^16
maxResListLen = 300;
```

So in `runFilterOnGpu`:

- `Sequence qSeq(par.maxSeqLen, ...)` allocates `maxLen + 1` bytes twice
  ([`Sequence.cpp#L13-L16`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/commons/Sequence.cpp#L13-L16)) — 128 KB, fixed.
- `profileBufferLength = par.maxSeqLen`, so `profile` is
  `malloc(alphabetSize * 65535)` ≈ 1.4 MB, fixed.
- The growth branch `if ((size_t)qSeq.L >= profileBufferLength)` **can never fire**
  for a query MMseqs2 accepted, because `--max-seq-len` is the same bound that
  admitted it.

A 5,000-residue query and a 62-residue query therefore allocate **identical**
host memory on this path. Query length affects the *inner loop trip count* — the
profile is filled `alphabetSize × L` times per query
([`ungappedprefilter.cpp#L188-L195`](https://github.com/soedinglab/MMseqs2/blob/8cc5ce367b5638c4306c2d7cfc652dd099a4643f/src/prefiltering/ungappedprefilter.cpp#L188-L195))
— and GPU work, hence runtime. Not memory.

*Inference, flagged as such:* the residue-based shard limit in
`workflows/mmseqs2-gpu.md` is therefore correctly a **runtime/throughput**
control. It should not be wired into the memory request.

---

## 6. Why CPU search uses more host memory than GPU search (507 vs 149 GB)

Four sourced mechanisms, all pointing the same way. They are additive.

1. **7 bytes/residue vs 1 byte/residue.** The k-mer index is the dominant CPU
   structure and is absent on the GPU path. "reducing memory usage to
   approximately 1 byte per residue in the target database instead of 7 bytes
   for the full k-mer index" (wiki, *GPU-accelerated search*); "from ~7 bytes to
   1 byte per residue" (Kallenborn et al. 2025, *Main*). This alone is a 7×
   reduction on the largest term.

2. **The k-mer index table itself.** `a^k * 8` bytes, independent of database
   content: for `a`=21, `k`=7 that is ~14.4 GB, plus `extendedMatrix` ~0.7 GB.
   Both are zero on the GPU path. (`Prefiltering.cpp#L1075`, `#L1090-L1093`.)

3. **The per-thread term, which the GPU path does not have.**
   `threads * (N/split) * ~50.5 B` of `QueryMatcher` scratch. A many-core CPU
   search multiplies target-database size by the thread count; the GPU search
   runs one query at a time with one set of buffers. On a 64-core node this term
   can exceed every other term combined.

4. **`fread` vs `mmap` at high sensitivity.** CPU `PRELOAD_MODE_AUTO` with
   `-s > 6.0` becomes `PRELOAD_MODE_FREAD`
   (`Prefiltering.cpp#L84-L90`), i.e. `malloc`+`fread` into anonymous memory
   (`DBReader.cpp#L442-L451`). The 507 GB may therefore be *genuinely reserved*
   memory in a way the 149 GB is not — the two numbers are not the same kind of
   quantity, even though `%M` reports both identically.

*Inference:* mechanisms 1-3 are structural and certain; (4) depends on the
sensitivity actually used in the CPU run, which I did not have. Worth checking
against the CPU command line before quoting the 507 GB as a reservable figure.

---

## What the sources do NOT settle

- **The 392 GB vs 149.4 GB gap.** If all four AF3 databases were touched into
  one process, RSS should approach the padded total, not 149.4 GB. The sources
  do not explain this. Two candidate explanations, both testable and neither
  sourced: (a) the databases are searched **sequentially, one `mmseqs` process
  per database**, so peak RSS is the *largest single* database rather than the
  sum — this is how ColabFold-style MSA generation is structured, and would make
  149.4 GB simply the biggest of the four; or (b) `touchMemory`'s
  `size > getTotalSystemMemory()` guard fired and skipped the sweep for some
  databases (`Util.cpp#L340-L343`). Checking whether one process opens all four
  DBs would settle it immediately, and it matters: under (a) the model should
  key on `max(db)`, not `sum(db)`.
- **No primary source gives a host-RAM formula for the GPU path.** The 1
  byte/residue figure is documented for the *index*, and streaming-from-host-RAM
  is documented, but no source states "host RAM ≈ padded DB size". That
  combination is my inference from the two sourced facts plus the `loadDb` call
  site.
- **Whether SLURM's cgroup accounting will OOM-kill on reclaimable page cache.**
  The MMseqs2 sources say nothing about cgroups. General Linux behaviour is that
  clean file-backed pages are reclaimed under pressure rather than triggering
  OOM, and that cgroup v2 `memory.current` (which SLURM reads) includes page
  cache — so a `--mem` request below the touched DB size would likely *work*
  while *reporting* alarming `MaxRSS`. **I could not source this for your SLURM
  configuration and it should be tested, not assumed**, since being wrong means
  job kills.
- **The MMseqs2 2017 Nature Biotechnology paper** (Steinegger & Söding,
  [doi:10.1038/nbt.3988](https://doi.org/10.1038/nbt.3988)) is paywalled and the
  bioRxiv preprint (079681) rate-limited during this research. Its memory claims
  are, however, restated in the User Guide formula quoted in §1, which is the
  same authors and is the version the implementation follows — so I do not think
  anything material is missing.
- **Linclust** (Steinegger & Söding 2018, Nat Commun) was not consulted in
  depth: it covers clustering, and clustering does not support GPUs
  ("Clustering does not support GPUs yet", wiki *GPU-accelerated search*), so it
  is not on this stage's path.
- **`Marv::Result` size** was not read, so the `results.reserve(maxResListLen)`
  term is not quantified. With ColabFold's `--max-seqs 10000` it is at most a
  few MB; ignoring it is safe but unmeasured.

---

## Recommended memory model

### GPU path (`--gpu 1`, padded DB) — what this stage uses

```
host_RAM  =  padded_target_db_bytes  +  C
```

with `C` ≈ 2-4 GB (CUDA context, DBWriter buffers, result accumulation), and
**no query-count term and no per-residue term**.

| Term | Justification |
| --- | --- |
| `padded_target_db_bytes` | DB is streamed to GPU *from host RAM* (wiki, *Database larger than GPU memory*; Kallenborn et al. 2025 *Methods → Database streaming*) and is touched into RSS unless `--db-load-mode 2` (`ungappedprefilter.cpp#L487-L488`). ≈1 byte/residue (wiki, *GPU-accelerated search*). |
| no query-count term | one query per loop iteration, buffers hoisted out of the loop (`ungappedprefilter.cpp#L47-L68`, `#L164`); "parallelize computation within a single query rather than across multiple queries" (wiki). |
| no per-residue term | query buffers pre-sized to `--max-seq-len` (2^16), not to actual length (`Parameters.cpp#L2354`, `ungappedprefilter.cpp#L47,L56-L57`). |
| `C` constant | shared-memory block is `~22 * maxSeqLen` ≈ 1.4 MB (`GpuUtil.h#L33-L38`); remainder is CUDA runtime — *my estimate, not sourced*. |

If the pipeline searches databases sequentially, use `max()` over databases
rather than `sum()` — **but confirm this first** (see *What the sources do NOT
settle*).

Add `--db-load-mode 2` to drop the touch sweep and let RSS reflect actual reads.
This is the single highest-value change available and is a one-flag experiment.

### CPU path — if you ever need to model it

```
host_RAM  =  7*R/split  +  a^k*8  +  threads*(N/split)*50.5  +  N*42  +  0.7GB
```

| Term | Justification |
| --- | --- |
| `7*R/split` | `residueSize` (`Prefiltering.cpp#L1073`); wiki `M = (7*N*L + 8a^k)`. |
| `a^k*8` | `indexTableSize` (`#L1075`); ~14.4 GB at `a`=21,`k`=7. |
| `threads*(N/split)*50.5` | `threadSize` (`#L1078-L1085`) with packed struct sizes; real allocation per `QueryMatcher.cpp#L40-L46`. **Scales with thread count** — do not omit. |
| `N*42` | `background` (`N*22`) + `dbReaderSize` (`#L1086`, `#L1095`). |
| `0.7 GB` | `extendedMatrix`, amino-acid queries only (`#L1090-L1093`). |
| no query term | the function's signature has none (`#L1067-L1070`); `qDbSize` is used only to cap split count (`#L326-L327`). |

Set `--split-memory-limit` explicitly to ~80% of the SLURM `--mem` (wiki:
"`--split-memory-limit` will be about 80% of the total memory required"), because
the default reads *physical node* memory and ignores the cgroup
(`Util.cpp#L584-L599`, `#L293-L315`).

### Bottom line

**A per-residue term in the memory model is not justified by any primary
source, and is contradicted by the allocation sites.** Keep residues as a
runtime/packing control; make memory a function of the target database (plus
threads on the CPU path).
