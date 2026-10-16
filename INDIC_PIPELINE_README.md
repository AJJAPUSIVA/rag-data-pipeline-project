# Indic dataset preprocessing extension (prototype)

This stand-alone, standard-library Python module complements the repository's retrieval examples; it **does not change** `retrieval_concepts.py`, `RAG/rag.py`, or `run_pipeline.py`.

## Features
- JSONL streaming, Unicode NFC normalization, whitespace cleanup
- Exact SHA-256 deduplication using a disk-backed SQLite table
- Exact normalized-text benchmark hash exclusion (must provide known hashes)
- Source and dataset-license metadata, per-language counts, output hash manifest
- Basic `unittest` coverage

## Quick start

```bash
python3 -m unittest discover -s tests -v
python3 indic_data_pipeline.py \
  --input examples/sample_indic.jsonl \
  --output build/indic_clean.jsonl \
  --manifest build/indic_manifest.json \
  --db build/indic_seen.sqlite \
  --source "Local synthetic demonstration" \
  --license "CC0-1.0"
```

The sample lines were written specifically as demonstration text. **Do not specify `CC0-1.0` for other sources without verifying their actual licence.** To exclude evaluation data, prepare a file containing the SHA-256 digest of each benchmark text after the same normalization, one digest per line; pass `--benchmark-hashes path/to/exclude.sha256`. Benchmark hashes, not benchmark raw text, are loaded by this module.

### Limitations
- This is **not** a production contamination firewall: exact normalized-text hashes miss paraphrases, substring overlap and near-duplicates; near-duplicate detection must be added and validated.
- Unicode NFC does not replace language-specific quality checks or native-speaker review.
- A dataset licence string is recorded but *not* automatically verified against an external registry.
- Output hashing streams in 1 MiB blocks instead of reading the output into memory.
- This example uses one local SQLite file and one worker, not a distributed system.
- If input has errors, processing stops; this prototype does not guarantee atomic output publication.
- Output and SQLite state are recreated on each run, so do not use the same paths concurrently.

## Suggested `.gitignore` addition

```gitignore
build/
__pycache__/
*.pyc
```
