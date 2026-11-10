#!/usr/bin/env python3
"""Build a context-preserving Ensu semantic pilot from the canonical SQLite DB.

The SQLite database is opened read-only. Each passage repeats its headword and
part of speech, and long content is split only at whitespace boundaries.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

DEFAULT_MAX_CHARS = 600


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return ""


def strings(value: Any) -> list[str]:
    """Extract readable strings from Wiktextract's mixed list/object fields."""
    out: list[str] = []
    if isinstance(value, str):
        s = clean(value)
        if s:
            out.append(s)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                s = clean(item)
                if s:
                    out.append(s)
            elif isinstance(item, dict):
                # Wiktextract examples commonly use text; gloss objects use gloss.
                for key in ("text", "gloss", "raw_gloss", "name", "ipa", "roman"):
                    s = clean(item.get(key))
                    if s:
                        out.append(s)
                        break
    elif isinstance(value, dict):
        for key in ("text", "gloss", "raw_gloss", "name", "ipa", "roman"):
            s = clean(value.get(key))
            if s:
                out.append(s)
    return out


def unique(items: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def field_values(record: dict, key: str) -> list[str]:
    return unique(strings(record.get(key)))


def get_headword(record: dict) -> str:
    return clean(record.get("word") or record.get("title") or record.get("lemma"))


def get_pos(record: dict) -> str:
    return clean(record.get("pos") or record.get("part_of_speech")) or "unspecified"


def get_pronunciations(record: dict) -> list[str]:
    values = []
    for sound in record.get("sounds", []) if isinstance(record.get("sounds"), list) else []:
        if isinstance(sound, dict):
            ipa = clean(sound.get("ipa"))
            if ipa:
                values.append(ipa)
    return unique(values)


def sense_label(sense: dict, index: int) -> str:
    tags = field_values(sense, "tags")
    topics = field_values(sense, "topics")
    labels = unique(tags + topics)
    prefix = f"sense {index + 1}"
    if labels:
        prefix += "; labels: " + ", ".join(labels[:8])
    return prefix


def sense_content(sense: dict) -> list[str]:
    parts = []
    glosses = field_values(sense, "glosses")
    raw_glosses = field_values(sense, "raw_glosses")
    definitions = unique(glosses + raw_glosses)
    if definitions:
        parts.append("Definition: " + "; ".join(definitions))

    examples = []
    raw_examples = sense.get("examples", [])
    if isinstance(raw_examples, list):
        for ex in raw_examples:
            if isinstance(ex, str):
                value = clean(ex)
            elif isinstance(ex, dict):
                value = clean(ex.get("text") or ex.get("example"))
                translation = clean(ex.get("translation"))
                if value and translation:
                    value += f" (translation: {translation})"
            else:
                value = ""
            if value:
                examples.append(value)
    if examples:
        parts.extend("Example: " + x for x in unique(examples))

    for key, label in (
        ("synonyms", "Synonyms"),
        ("antonyms", "Antonyms"),
        ("related", "Related terms"),
        ("derived", "Derived terms"),
        ("holonyms", "Holonyms"),
        ("meronyms", "Meronyms"),
        ("translations", "Translations"),
    ):
        values = field_values(sense, key)
        if values:
            parts.append(label + ": " + "; ".join(values))

    return parts


def split_text(text: str, limit: int) -> list[str]:
    """Split at whitespace; prefer sentence/semicolon boundaries where possible."""
    text = clean(text)
    if len(text) <= limit:
        return [text] if text else []

    chunks = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit + 1]
        # Prefer a sentence or clause boundary in the latter 60% of the window.
        cut = -1
        floor = int(limit * 0.60)
        for match in re.finditer(r"[.!?;]\s+", window):
            if match.end() >= floor:
                cut = match.end()
        if cut < 0:
            # Otherwise split at the last whitespace within the limit.
            cut = window.rfind(" ", 0, limit + 1)
        if cut <= 0:
            # A single unbroken token is exceptionally long; preserve it intact.
            next_space = remaining.find(" ", limit)
            if next_space < 0:
                chunks.append(remaining)
                remaining = ""
                break
            cut = next_space
        piece = remaining[:cut].strip()
        if piece:
            chunks.append(piece)
        remaining = remaining[cut:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


def make_passages(record: dict, max_chars: int) -> list[dict]:
    word = get_headword(record)
    if not word:
        return []
    pos = get_pos(record)
    pronunciations = get_pronunciations(record)
    prefix = f"Headword: {word}. Part of speech: {pos}."
    if pronunciations:
        prefix += " Pronunciation: " + ", ".join(pronunciations[:6]) + "."

    senses = record.get("senses", [])
    if not isinstance(senses, list) or not senses:
        senses = [{}]

    passages = []
    for index, sense in enumerate(senses):
        if not isinstance(sense, dict):
            continue
        label = sense_label(sense, index)
        content_parts = sense_content(sense)
        if not content_parts:
            continue

        # Keep each semantic unit whole when possible. Long units are split
        # into whitespace-safe fragments, with context repeated in every chunk.
        context = prefix + " " + label + ". "
        budget = max(80, max_chars - len(context))
        units = []
        for part in content_parts:
            units.extend(split_text(part, budget))

        current = []
        current_len = 0
        for unit in units:
            addition = len(unit) + (1 if current else 0)
            if current and current_len + addition > budget:
                text = context + " ".join(current)
                passages.append({
                    "title": word,
                    "text": text,
                    "section": f"{pos}; {label}; passage {len(passages) + 1}",
                })
                current = []
                current_len = 0
            current.append(unit)
            current_len += len(unit) + (1 if len(current) > 1 else 0)

        if current:
            text = context + " ".join(current)
            passages.append({
                "title": word,
                "text": text,
                "section": f"{pos}; {label}; passage {len(passages) + 1}",
            })

    return passages


def sample_ids(conn: sqlite3.Connection, count: int) -> list[int]:
    total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    if total == 0:
        return []
    count = min(count, total)
    # Stable, evenly distributed sample across the complete import.
    ids = []
    for i in range(count):
        offset = (i * (total - 1)) // max(1, count - 1) if count > 1 else total // 2
        row = conn.execute(
            "SELECT id FROM entries ORDER BY id LIMIT 1 OFFSET ?", (offset,)
        ).fetchone()
        if row:
            ids.append(row[0])
    return total, ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", "--db", dest="database", default="build/dictionary.sqlite3")
    parser.add_argument("--entries", type=int, default=1000)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--output-dir", "--out", dest="output_dir", default="build/semantic-pilot")
    args = parser.parse_args()

    db_path = Path(args.database).resolve()
    if not db_path.is_file():
        raise SystemExit(f"Database not found: {db_path}")
    if args.entries < 1 or args.max_chars < 200:
        raise SystemExit("--entries must be positive and --max-chars must be at least 200")

    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        state = conn.execute(
            "SELECT complete FROM import_state ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if not state or state["complete"] != 1:
            raise SystemExit("Dictionary import is not marked complete.")

        total, ids = sample_ids(conn, args.entries)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "pilot.jsonl"

        entries_with_passages = 0
        entries_without_passages = 0
        passage_count = 0
        max_observed = 0

        with jsonl_path.open("w", encoding="utf-8", newline="\n") as out:
            for entry_id in ids:
                row = conn.execute(
                    "SELECT raw_json FROM entries WHERE id = ?", (entry_id,)
                ).fetchone()
                if not row:
                    entries_without_passages += 1
                    continue
                try:
                    record = json.loads(row["raw_json"])
                except (TypeError, json.JSONDecodeError):
                    entries_without_passages += 1
                    continue

                passages = make_passages(record, args.max_chars)
                if passages:
                    entries_with_passages += 1
                else:
                    entries_without_passages += 1
                for passage in passages:
                    max_observed = max(max_observed, len(passage["text"]))
                    out.write(json.dumps(passage, ensure_ascii=False) + "\n")
                    passage_count += 1

        report = {
            "source_database": str(args.database),
            "source_english_entries": total,
            "requested_sample_entries": args.entries,
            "sampled_ids": len(ids),
            "entries_with_passages": entries_with_passages,
            "entries_without_passages": entries_without_passages,
            "semantic_passages": passage_count,
            "max_chars": args.max_chars,
            "max_observed_passage_chars": max_observed,
            "all_passages_repeat_headword_and_pos": True,
            "splitting": "sentence_or_whitespace_boundaries",
            "output_jsonl": str(jsonl_path),
            "source_database_modified": False,
            "note": "Full raw Wiktextract records remain in the original SQLite database.",
        }
        (output_dir / "pilot-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
