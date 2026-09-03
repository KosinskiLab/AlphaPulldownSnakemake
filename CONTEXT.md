# Domain language

- **Feature request**: one named protein sequence requiring an AlphaFold 3 feature artifact.
- **Feature batch**: an ordered collection of missing protein feature requests handled as one operation; its deep interface owns sequence deduplication and query chunking.
- **Database identifier**: the configured immutable identity of one MMseqs2 database build.
- **Feature artifact**: the standard per-protein AlphaFold 3 JSON consumed by structure inference.
- **Cache hit**: an artifact that AlphaPulldown validates against the sequence, MMseqs2 settings, and database identifiers.
