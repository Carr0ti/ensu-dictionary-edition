#!/usr/bin/env python3
"""Lossless English Wiktextract JSONL -> SQLite importer.

The original JSON object is stored verbatim (as decoded JSON text) in raw_json.
The additional columns are indexes for lookup and validation; they do not replace
or truncate the source record.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path
from typing import Any, Iterator

DEFAULT_URL = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=FILE;

CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY,
    word TEXT NOT NULL,
    normalized_word TEXT NOT NULL,
    lang TEXT,
    lang_code TEXT,
    pos TEXT,
    raw_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entries_normalized_word ON entries(normalized_word);
CREATE INDEX IF NOT EXISTS idx_entries_word ON entries(word);
CREATE INDEX IF NOT EXISTS idx_entries_lang_code ON entries(lang_code);
CREATE INDEX IF NOT EXISTS idx_entries_pos ON entries(pos);

CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
    word, raw_json, content='entries', content_rowid='id'
);
"""

def is_english(record: dict[str, Any]) -> bool:
    """Accept records explicitly marked English by Wiktextract."""
    lang_code = str(record.get("lang_code") or "").strip().lower()
    lang = str(record.get("lang") or "").strip().lower()
    return lang_code == "en" or lang == "english"

def normalized(word: str) -> str:
    return " ".join(word.casefold().split())

def iter_records(stream) -> Iterator[dict[str, Any]]:
    for line_no, line in enumerate(stream, 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on source line {line_no}: {exc}") from exc
        if isinstance(obj, dict):
            yield obj

def open_source(source: str):
    if source.startswith("http://") or source.startswith("https://"):
        request = urllib.request.Request(source, headers={"User-Agent": "Ensu-Dictionary-Edition/0.1"})
        response = urllib.request.urlopen(request, timeout=60)
        return gzip.GzipFile(fileobj=response, mode="rb"), response
    path = Path(source)
    raw = path.open("rb")
    if source.endswith(".gz"):
        return gzip.GzipFile(fileobj=raw, mode="rb"), raw
    return raw, raw

def import_data(source: str, output: Path, limit: int | None, batch_size: int) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    db = sqlite3.connect(output)
    db.executescript(SCHEMA)
    inserted = 0
    seen_source = 0
    batch = []
    stream, owner = open_source(source)
    try:
        import io
        text_stream = io.TextIOWrapper(stream, encoding="utf-8", errors="strict")
        for record in iter_records(text_stream):
            seen_source += 1
            if not is_english(record):
                continue
            word = str(record.get("word") or "").strip()
            if not word:
                continue
            raw_json = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            batch.append((
                word,
                normalized(word),
                record.get("lang"),
                record.get("lang_code"),
                record.get("pos"),
                raw_json,
            ))
            if len(batch) >= batch_size:
                db.executemany(
                    """INSERT INTO entries(word, normalized_word, lang, lang_code, pos, raw_json)
                       VALUES (?, ?, ?, ?, ?, ?)""", batch
                )
                db.commit()
                inserted += len(batch)
                batch.clear()
                if inserted % (batch_size * 10) == 0:
                    print(f"English records stored: {inserted:,}", flush=True)
            if limit is not None and inserted + len(batch) >= limit:
                break
        if batch:
            db.executemany(
                """INSERT INTO entries(word, normalized_word, lang, lang_code, pos, raw_json)
                   VALUES (?, ?, ?, ?, ?, ?)""", batch
            )
            inserted += len(batch)
            db.commit()
        # Populate FTS after the complete import. This is derived data; raw_json remains canonical.
        db.execute("INSERT INTO entries_fts(entries_fts) VALUES('rebuild')")
        db.commit()
        return {"source_records_seen": seen_source, "english_records_stored": inserted}
    finally:
        try:
            stream.close()
        finally:
            if owner is not stream:
                owner.close()
            db.close()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_URL, help="HTTPS URL or local .jsonl/.jsonl.gz file")
    parser.add_argument("--output", default="build/dictionary.sqlite3")
    parser.add_argument("--limit", type=int, default=None, help="Optional English-record limit for smoke tests")
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    stats = import_data(args.source, Path(args.output), args.limit, args.batch_size)
    print(json.dumps(stats, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
