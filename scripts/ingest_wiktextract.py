#!/usr/bin/env python3
"""Lossless English Wiktextract JSONL -> resumable SQLite importer.

The original Wiktextract record is retained in raw_json. Progress is stored in
the same SQLite database and committed atomically with each batch.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
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

CREATE TABLE IF NOT EXISTS import_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
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
        request = urllib.request.Request(
            source, headers={"User-Agent": "Ensu-Dictionary-Edition/0.2"}
        )
        response = urllib.request.urlopen(request, timeout=60)
        return gzip.GzipFile(fileobj=response, mode="rb"), response
    path = Path(source)
    raw = path.open("rb")
    if source.endswith(".gz"):
        return gzip.GzipFile(fileobj=raw, mode="rb"), raw
    return raw, raw

def _set_state(db: sqlite3.Connection, source: str, offset: int,
               source_records: int, english_records: int, complete: bool) -> None:
    values = {
        "source": source,
        "source_offset": str(offset),
        "source_records_seen": str(source_records),
        "english_records_stored": str(english_records),
        "complete": "1" if complete else "0",
    }
    db.executemany(
        "INSERT INTO import_state(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        values.items(),
    )

def _get_state(db: sqlite3.Connection) -> dict[str, str]:
    return dict(db.execute("SELECT key, value FROM import_state"))

def import_data(
    source: str,
    output: Path,
    limit: int | None,
    batch_size: int,
    resume: bool = False,
    checkpoint_source_records: int = 100_000,
    stop_after_source_records: int | None = None,
) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not resume:
        output.unlink()
        for suffix in ("-wal", "-shm"):
            Path(str(output) + suffix).unlink(missing_ok=True)

    db = sqlite3.connect(output)
    db.executescript(SCHEMA)

    state = _get_state(db)
    if resume and state:
        if state.get("source") != source:
            db.close()
            raise ValueError(
                "Checkpoint source does not match this run. Use the same source "
                "or start a fresh import without --resume."
            )
        if state.get("complete") == "1":
            stats = {
                "source_records_seen": int(state["source_records_seen"]),
                "english_records_stored": int(state["english_records_stored"]),
            }
            db.close()
            print("Import is already complete; keeping the existing database.")
            return stats
        start_offset = int(state.get("source_offset", "0"))
        seen_source = int(state.get("source_records_seen", "0"))
        inserted = int(state.get("english_records_stored", "0"))
        print(
            f"Resuming from checkpoint: {seen_source:,} source records scanned, "
            f"{inserted:,} English records stored.",
            flush=True,
        )
    else:
        start_offset = 0
        seen_source = 0
        inserted = 0
        _set_state(db, source, 0, 0, 0, False)
        db.commit()

    batch = []
    stream, owner = open_source(source)
    try:
        # GzipFile.seek() replays decompression to the saved uncompressed offset.
        # This preserves committed database work, although the compressed source
        # must still be read again up to that point after a runner restart.
        if start_offset:
            stream.seek(start_offset)

        last_checkpoint_seen = seen_source
        for line in stream:
            if not line.strip():
                continue
            seen_source += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on source line {seen_source}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                continue

            if is_english(record):
                word = str(record.get("word") or "").strip()
                if word:
                    raw_json = json.dumps(
                        record, ensure_ascii=False, separators=(",", ":")
                    )
                    batch.append((
                        word,
                        normalized(word),
                        record.get("lang"),
                        record.get("lang_code"),
                        record.get("pos"),
                        raw_json,
                    ))

            hit_limit = limit is not None and inserted + len(batch) >= limit
            hit_test_stop = (
                stop_after_source_records is not None
                and seen_source - int(state.get("source_records_seen", "0")) >= stop_after_source_records
            )
            should_checkpoint = (
                len(batch) >= batch_size
                or seen_source - last_checkpoint_seen >= checkpoint_source_records
                or hit_limit
                or hit_test_stop
            )
            if should_checkpoint:
                if batch:
                    remaining = batch
                    if limit is not None:
                        remaining = remaining[: max(0, limit - inserted)]
                    db.executemany(
                        """INSERT INTO entries(word, normalized_word, lang, lang_code, pos, raw_json)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        remaining,
                    )
                    inserted += len(remaining)
                    batch.clear()
                offset = stream.tell()
                _set_state(
                    db, source, offset, seen_source, inserted,
                    complete=bool(hit_limit),
                )
                db.commit()
                last_checkpoint_seen = seen_source
                if inserted and inserted % (batch_size * 10) == 0:
                    print(f"English records stored: {inserted:,}", flush=True)
                if hit_limit or hit_test_stop:
                    break

        else:
            # Natural end of source: commit any remaining records and mark complete.
            if batch:
                remaining = batch
                if limit is not None:
                    remaining = remaining[: max(0, limit - inserted)]
                db.executemany(
                    """INSERT INTO entries(word, normalized_word, lang, lang_code, pos, raw_json)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    remaining,
                )
                inserted += len(remaining)
                batch.clear()
            _set_state(db, source, stream.tell(), seen_source, inserted, complete=True)
            db.commit()

        # Build FTS only when the import is complete. Interrupted runs can
        # resume without spending time rebuilding an index that will change.
        final_state = _get_state(db)
        if final_state.get("complete") == "1":
            db.execute("INSERT INTO entries_fts(entries_fts) VALUES('rebuild')")
            db.commit()
        return {"source_records_seen": seen_source, "english_records_stored": inserted}
    finally:
        try:
            stream.close()
        finally:
            if owner is not stream:
                owner.close()
            # Flush WAL into the main database before GitHub Actions caches it.
            try:
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                db.close()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_URL,
                        help="HTTPS URL or local .jsonl/.jsonl.gz file")
    parser.add_argument("--output", default="build/dictionary.sqlite3")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional maximum English records (for smoke tests)")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--checkpoint-source-records", type=int, default=100_000,
                        help="Save progress after this many source records")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from the checkpoint in the existing database")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.batch_size < 1 or args.checkpoint_source_records < 1:
        parser.error("--batch-size and --checkpoint-source-records must be positive")
    stats = import_data(
        args.source, Path(args.output), args.limit, args.batch_size,
        resume=args.resume,
        checkpoint_source_records=args.checkpoint_source_records,
    )
    print(json.dumps(stats, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
