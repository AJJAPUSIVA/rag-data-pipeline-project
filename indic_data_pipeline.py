"""Streaming JSONL Indic-text preparation with licence checks and benchmark exclusions.

Standard library only. Intended as a starter for a larger auditable data pipeline.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path


def normalize(text: str) -> str:
    # NFC preserves Indic combining-character distinctions better than aggressive NFKC.
    return ' '.join(unicodedata.normalize('NFC', text).split())


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def process(source: Path, output: Path, manifest: Path, db_path: Path,
            source_name: str, license_id: str, benchmark_hashes: Path | None = None) -> dict:
    if not license_id.strip() or license_id.lower() in {'unknown', 'none', 'tbd'}:
        raise ValueError('Record and verify the dataset licence before processing.')
    if not source_name.strip():
        raise ValueError('Dataset source must be recorded.')
    source, output, manifest, db_path = map(Path, (source, output, manifest, db_path))
    if len({p.resolve() for p in (source, output, manifest, db_path)}) != 4:
        raise ValueError('Input, output, manifest, and database paths must differ.')
    if benchmark_hashes and Path(benchmark_hashes).resolve() in {source.resolve(), output.resolve(), manifest.resolve(), db_path.resolve()}:
        raise ValueError('Benchmark hash file must be separate from all other files.')
    excluded = set()
    if benchmark_hashes:
        with Path(benchmark_hashes).open(encoding='utf-8') as handle:
            excluded = {line.strip().lower() for line in handle if line.strip() and not line.lstrip().startswith('#')}
        if any(len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in excluded):
            raise ValueError('Each benchmark exclusion must be a SHA-256 hex digest.')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    by_language = Counter()
    with sqlite3.connect(db_path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS seen (hash TEXT PRIMARY KEY)')
        # Each invocation starts with an empty dedup database; no cross-run contamination.
        db.execute('DELETE FROM seen')
        with source.open(encoding='utf-8') as src, output.open('w', encoding='utf-8') as dest:
            for line_number, line in enumerate(src, 1):
                stats['read'] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f'Malformed JSON at input line {line_number}') from e
                if not isinstance(record, dict) or not isinstance(record.get('text'), str):
                    raise ValueError(f'Expected object with string text at line {line_number}')
                text = normalize(record['text'])
                if not text:
                    stats['empty'] += 1
                    continue
                digest = sha256(text)
                if digest in excluded:
                    stats['benchmark_excluded'] += 1
                    continue
                cursor = db.execute('INSERT OR IGNORE INTO seen(hash) VALUES (?)', (digest,))
                if cursor.rowcount == 0:
                    stats['duplicates'] += 1
                    continue
                record['text'] = text
                record['source'] = source_name
                record['license'] = license_id
                dest.write(json.dumps(record, ensure_ascii=False) + '\n')
                stats['accepted'] += 1
                by_language[str(record.get('language', 'unknown'))] += 1
        db.commit()
    output_hasher = hashlib.sha256()
    with output.open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            output_hasher.update(block)
    result = {
        'source': source_name,
        'license': license_id,
        'input_file': str(source),
        'output_file': str(output),
        'output_sha256': output_hasher.hexdigest(),
        'normalization': 'Unicode NFC + whitespace collapse',
        'deduplication': 'exact SHA-256 after normalization; disk-backed SQLite',
        'benchmark_exclusions_file': str(benchmark_hashes) if benchmark_hashes else None,
        'counts': {name: stats[name] for name in ('read', 'accepted', 'empty', 'duplicates', 'benchmark_excluded')},
        'language_counts': dict(sorted(by_language.items())),
        'limitations': 'Not legal approval, script detection, near-duplicate detection, or exhaustive benchmark contamination protection.'
    }
    manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--source', required=True, help='Dataset name/URL for provenance')
    parser.add_argument('--license', required=True, help='Verified source dataset licence identifier')
    parser.add_argument('--benchmark-hashes', type=Path, help='One normalized-text SHA-256 per line')
    args = parser.parse_args()
    print(json.dumps(process(args.input, args.output, args.manifest, args.db,
                             args.source, args.license, args.benchmark_hashes), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
