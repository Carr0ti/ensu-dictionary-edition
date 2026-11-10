#!/usr/bin/env python3
"""Prepare a deterministic semantic-search pilot from the lossless dictionary DB.

This stage does NOT embed data or modify the source SQLite database. It exports
up to N evenly sampled English Wiktextract records as Ensu-shaped passages.
The complete raw records remain in dictionary.sqlite3.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

MAX_CHARS = 600


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.replace("\x00", " ").split())
    return " ".join(str(value).split())


def list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, str):
            text = clean(item)
        elif isinstance(item, dict):
            text = clean(item.get("text") or item.get("gloss") or item.get("word"))
        else:
            text = ""
        if text and text not in result:
            result.append(text)
    return result


def split_text(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Split at whitespace without dropping content."""
    text = clean(text)
    if not text:
        return []
    parts: list[str] = []
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit + 1)
        if cut < max(1, limit // 2):
            cut = limit
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def record_passages(record: dict[str, Any]) -> list[dict[str, str]]:
    word = clean(record.get("word"))
    pos = clean(record.get("pos")) or "unknown part of speech"
    if not word:
        return []

    sounds = record.get("sounds") if isinstance(record.get("sounds"), list) else []
    ipa = []
    for sound in sounds:
        if isinstance(sound, dict):
            value = clean(sound.get("ipa"))
            if value and value not in ipa:
                ipa.append(value)
    pronunciation = f"Pronunciation: {', '.join(ipa[:4])}. " if ipa else ""

    senses = record.get("senses")
    if not isinstance(senses, list) or not senses:
        senses = [{"glosses": list_of_strings(record.get("glosses"))}]

    rows: list[dict[str, str]] = []
    for sense_no, sense in enumerate(senses, 1):
        if not isinstance(sense, dict):
            continue
        glosses = list_of_strings(sense.get("glosses"))
        if not glosses:
            glosses = list_of_strings(sense.get("raw_glosses"))
        tags = list_of_strings(sense.get("tags"))
        topics = list_of_strings(sense.get("topics"))
        examples = sense.get("examples") if isinstance(sense.get("examples"), list) else []

        sections = [
            f"Headword: {word}.",
            f"Part of speech: {pos}.",
        ]
        if pronunciation:
            sections.append(pronunciation.strip())
        if tags:
            sections.append("Labels: " + ", ".join(tags) + ".")
        if topics:
            sections.append("Topics: " + ", ".join(topics) + ".")
        if glosses:
            sections.append("Definition: " + "; ".join(glosses))

        for example in examples:
            if not isinstance(example, dict):
                continue
            example_text = clean(example.get("text"))
            translation = clean(example.get("translation"))
            if example_text:
                line = "Example: " + example_text
                if translation:
                    line += " (translation: " + translation + ")"
                sections.append(line + ".")

        full = " ".join(part for part in sections if part)
        chunks = split_text(full)
        for chunk_no, chunk in enumerate(chunks, 1):
            rows.append({
                "title": word,
                "text": chunk,
                "section": f"{pos}; sense {sense_no}; passage {chunk_no}",
            })

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="build/dictionary.sqlite3")
    parser.add_argument("--entries", type=int, default=1000)
    parser.add_argument("--out", default="build/semantic-pilot")
    args = parser.parse_args()

    if args.entries < 1:
        parser.error("--entries must be positive")

    db_path = Path(args.db)
    if not db_path.is_file():
        raise SystemExit(f"Database not found: {db_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "pilot.jsonl"
    report_path = out_dir / "pilot-report.json"

    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    total = db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    if total == 0:
        raise SystemExit("The dictionary database has no entries")

    # Deterministic, evenly spaced IDs avoid a slow ORDER BY RANDOM() over
    # the 1.5-million-row source table and avoid sampling only A-words.
    sample_count = min(args.entries, total)
    ids = sorted({
        max(1, min(total, 1 + (i * (total - 1) // max(1, sample_count - 1))))
        for i in range(sample_count)
    })

    written_entries = 0
    written_passages = 0
    empty_records = 0
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for entry_id in ids:
            row = db.execute(
                "SELECT raw_json FROM entries WHERE id = ?", (entry_id,)
            ).fetchone()
            if row is None:
                continue
            try:
                record = json.loads(row[0])
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            passages = record_passages(record)
            if not passages:
                empty_records += 1
                continue
            written_entries += 1
            for passage in passages:
                handle.write(json.dumps(
                    passage, ensure_ascii=False, separators=(",", ":")
                ) + "\n")
                written_passages += 1

    db.close()

    report = {
        "source_database": str(db_path),
        "source_english_entries": total,
        "requested_sample_entries": args.entries,
        "sampled_ids": len(ids),
        "entries_with_passages": written_entries,
        "entries_without_passages": empty_records,
        "semantic_passages": written_passages,
        "max_chars": MAX_CHARS,
        "output_jsonl": str(jsonl_path),
        "source_database_modified": False,
        "note": "Full raw Wiktextract records remain in the original SQLite database."
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if written_entries == 0 or written_passages == 0:
        raise SystemExit("Pilot export produced no usable passages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
