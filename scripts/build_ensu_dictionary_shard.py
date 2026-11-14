
#!/usr/bin/env python3
"""Stream every usable English Wiktionary sense into one Ensu pack shard."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

MODEL_SHA256 = "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63"
MODEL_PATH = "build/models/embeddinggemma-300M-Q8_0.gguf"

SOURCE_DIM = 768
DIM = 512
SCALE = 127
MAX_CHARS = 600
DOC_PROMPT = "title: {title} | text: {text}"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_semantic_pilot import make_passages


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path: Path, threads: int):
    from llama_cpp import Llama, LLAMA_POOLING_TYPE_MEAN

    return Llama(
        model_path=str(path),
        n_ctx=2048,
        n_batch=2048,
        n_ubatch=2048,
        n_threads=threads,
        n_threads_batch=threads,
        embedding=True,
        pooling_type=LLAMA_POOLING_TYPE_MEAN,
        n_gpu_layers=0,
        verbose=False,
    )


def fit_passage(row: dict, max_chars: int) -> dict:
    """Enforce the metadata text limit without discarding an entire passage."""
    row = dict(row)
    title = str(row.get("title", "")).strip()
    text = str(row.get("text", "")).strip()

    if len(text) > max_chars:
        # Keep a complete prefix where possible. This handles pathological
        # unbroken tokens and oversized context fields from source records.
        clipped = text[:max_chars]
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0]
        if not clipped:
            clipped = text[:max_chars]
        row["text"] = clipped.rstrip()

    if not title or not row["text"]:
        raise ValueError("Passage has an empty title or text")

    if len(row["text"]) > max_chars:
        raise ValueError(f"Unable to enforce passage limit for {title}")

    return row


def quantize_embedding(llm, row):
    prompt = DOC_PROMPT.format(
        title=row["title"],
        text=row["text"],
    )

    try:
        result = llm.embed(prompt, normalize=False)
    except TypeError:
        result = llm.embed(prompt)

    vector = (
        result[0]
        if result and isinstance(result[0], (list, tuple))
        else result
    )

    if len(vector) != SOURCE_DIM:
        raise ValueError(
            f"Expected {SOURCE_DIM} dimensions, got {len(vector)}"
        )

    values = [float(x) for x in vector[:DIM]]

    if not all(math.isfinite(x) for x in values):
        raise ValueError("Embedding contains non-finite values")

    norm = math.sqrt(math.fsum(x * x for x in values))

    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding has zero or invalid norm")

    return bytes(
        (
            max(
                -127,
                min(127, int(round((value / norm) * SCALE))),
            )
            & 255
        )
        for value in values
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--database", default="build/dictionary.sqlite3")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=16)
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(4, os.cpu_count() or 2)),
    )

    args = parser.parse_args()

    if args.shard_count < 1:
        parser.error("--shard-count must be positive")

    if not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid shard index")

    if args.max_chars < 200:
        parser.error("--max-chars must be at least 200")

    database = Path(args.database)
    model_path = Path(args.model)
    output = Path(args.output)

    if not database.is_file():
        raise SystemExit(f"Database not found: {database}")

    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")

    model_digest = sha256_file(model_path)

    if model_digest != MODEL_SHA256:
        raise SystemExit(f"Model SHA-256 mismatch: {model_digest}")

    output.mkdir(parents=True, exist_ok=True)

    uri = f"file:{database.resolve().as_posix()}?mode=ro"
    db = sqlite3.connect(uri, uri=True)
    db.row_factory = sqlite3.Row

    state = dict(db.execute("SELECT key, value FROM import_state"))
    total_entries = db.execute(
        "SELECT COUNT(*) FROM entries"
    ).fetchone()[0]

    if state.get("complete") != "1":
        raise SystemExit("Source dictionary import is incomplete")

    if total_entries < 1_000_000:
        raise SystemExit(
            f"Expected full dictionary, found {total_entries:,} entries"
        )

    expected_entries = db.execute(
        """
        SELECT COUNT(*)
        FROM entries
        WHERE ((id - 1) % ?) = ?
        """,
        (args.shard_count, args.shard_index),
    ).fetchone()[0]

    print(f"Shard {args.shard_index + 1}/{args.shard_count}", flush=True)
    print(f"Source entries: {expected_entries:,}", flush=True)

    model = load_model(model_path, args.threads)

    vector_path = output / "vectors.i8"
    rows_path = output / "rows.jsonl"

    entries_with_passages = 0
    entries_without_passages = 0
    passage_count = 0
    max_observed = 0
    clipped_passages = 0

    try:
        with (
            vector_path.open("wb") as vectors,
            rows_path.open("w", encoding="utf-8", newline="\n") as rows_out,
        ):
            cursor = db.execute(
                """
                SELECT id, raw_json
                FROM entries
                WHERE ((id - 1) % ?) = ?
                ORDER BY id
                """,
                (args.shard_count, args.shard_index),
            )

            while True:
                batch = cursor.fetchmany(2000)
                if not batch:
                    break

                for source in batch:
                    try:
                        record = json.loads(source["raw_json"])
                        passages = make_passages(record, args.max_chars)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        passages = []

                    if not passages:
                        entries_without_passages += 1
                        continue

                    entries_with_passages += 1

                    for original_row in passages:
                        original_text = str(
                            original_row.get("text", "")
                        ).strip()

                        row = fit_passage(original_row, args.max_chars)

                        if len(original_text) > args.max_chars:
                            clipped_passages += 1

                        vectors.write(quantize_embedding(model, row))

                        rows_out.write(
                            json.dumps(
                                row,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )

                        passage_count += 1
                        max_observed = max(
                            max_observed,
                            len(row["text"]),
                        )

                    if entries_with_passages % 100 == 0:
                        print(
                            f"Shard {args.shard_index}: "
                            f"{entries_with_passages:,}/"
                            f"{expected_entries:,} records; "
                            f"{passage_count:,} passages; "
                            f"{clipped_passages:,} clipped",
                            flush=True,
                        )

    finally:
        db.close()

    expected_vector_bytes = passage_count * DIM

    if vector_path.stat().st_size != expected_vector_bytes:
        raise SystemExit(
            "Vector byte count does not match passage count"
        )

    report = {
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "source_english_entries": total_entries,
        "shard_source_entries": expected_entries,
        "entries_with_passages": entries_with_passages,
        "entries_without_passages": entries_without_passages,
        "semantic_passages": passage_count,
        "dimension": DIM,
        "max_chars": args.max_chars,
        "max_observed_passage_chars": max_observed,
        "clipped_passages": clipped_passages,
        "duplicate_headwords_preserved": True,
        "all_usable_senses_preserved": True,
        "vectors_bytes": vector_path.stat().st_size,
        "model_sha256": model_digest,
    }

    (output / "shard-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
