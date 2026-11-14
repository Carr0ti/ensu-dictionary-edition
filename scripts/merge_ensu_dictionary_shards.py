
#!/usr/bin/env python3
"""Merge streamed dictionary shards into an Ensu-compatible pack."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import zstandard as zstd

DIM = 512
ROWS_PER_BLOCK = 128
MAX_CHARS = 600

MODEL_ID = (
    "ggml-org/embeddinggemma-300M-GGUF:"
    "embeddinggemma-300M-Q8_0.gguf"
)
MODEL_SHA256 = (
    "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63"
)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for chunk in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shards-dir",
        default="build/shards",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--output",
        default=(
            "build/semantic-pack/dictionary/"
            "dictionary-full-v1"
        ),
    )

    args = parser.parse_args()

    root = Path(args.shards_dir)
    output = Path(args.output)

    output.mkdir(parents=True, exist_ok=True)

    shard_dirs = [
        root / f"ensu-dictionary-shard-{i}"
        for i in range(args.shard_count)
    ]

    for directory in shard_dirs:
        for filename in (
            "vectors.i8",
            "rows.jsonl",
            "shard-report.json",
        ):
            path = directory / filename

            if not path.is_file():
                raise SystemExit(
                    f"Missing required shard file: {path}"
                )

    reports = [
        json.loads(
            (directory / "shard-report.json").read_text(
                encoding="utf-8"
            )
        )
        for directory in shard_dirs
    ]

    for index, report in enumerate(reports):
        if report["shard_index"] != index:
            raise SystemExit("Shard index mismatch")

        if report["shard_count"] != args.shard_count:
            raise SystemExit("Shard count mismatch")

        if report["model_sha256"] != MODEL_SHA256:
            raise SystemExit("Embedding model mismatch")

    source_count = reports[0]["source_english_entries"]

    if any(
        report["source_english_entries"] != source_count
        for report in reports
    ):
        raise SystemExit(
            "Shards were built from different source databases"
        )

    expected_source_records = (
        source_count + args.shard_count - 1
    ) // args.shard_count

    actual_source_records = sum(
        report["shard_source_entries"]
        for report in reports
    )

    if actual_source_records != source_count:
        raise SystemExit(
            "Shard source-record coverage is incomplete"
        )

    total_passages = sum(
        report["semantic_passages"]
        for report in reports
    )

    if total_passages <= 0:
        raise SystemExit("No semantic passages were produced")

    vectors_path = output / "vectors.i8"

    print("Merging vector files...", flush=True)

    with vectors_path.open("wb") as target:
        for directory, report in zip(shard_dirs, reports):
            source = directory / "vectors.i8"

            expected_bytes = (
                report["semantic_passages"] * DIM
            )

            if source.stat().st_size != expected_bytes:
                raise SystemExit(
                    f"Invalid vector size in {directory}"
                )

            with source.open("rb") as source_file:
                while True:
                    chunk = source_file.read(
                        8 * 1024 * 1024
                    )

                    if not chunk:
                        break

                    target.write(chunk)

    if vectors_path.stat().st_size != total_passages * DIM:
        raise SystemExit("Merged vector count is incorrect")

    print("Merging metadata blocks...", flush=True)

    compressor = zstd.ZstdCompressor(level=9)
    offsets = [0]
    block = []
    rows_seen = 0

    metadata_path = output / "meta.zst"

    with metadata_path.open("wb") as metadata:
        def flush_block():
            if not block:
                return

            payload = "".join(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ) + "\n"
                for row in block
            ).encode("utf-8")

            metadata.write(compressor.compress(payload))
            offsets.append(metadata.tell())
            block.clear()

        for directory in shard_dirs:
            rows_path = directory / "rows.jsonl"

            with rows_path.open(
                encoding="utf-8"
            ) as source:
                for line in source:
                    if not line.strip():
                        continue

                    row = json.loads(line)

                    title = str(
                        row.get("title", "")
                    ).strip()

                    text = str(
                        row.get("text", "")
                    ).strip()

                    if not title or not text:
                        raise SystemExit(
                            f"Invalid metadata row in {directory}"
                        )

                    if len(text) > MAX_CHARS:
                        raise SystemExit(
                            f"Passage exceeds {MAX_CHARS} characters"
                        )

                    block.append(row)
                    rows_seen += 1

                    if len(block) == ROWS_PER_BLOCK:
                        flush_block()

        flush_block()

    if rows_seen != total_passages:
        raise SystemExit(
            f"Metadata count mismatch: "
            f"{rows_seen:,} != {total_passages:,}"
        )

    offsets_path = output / "meta.offsets"

    offsets_path.write_bytes(
        b"".join(
            struct.pack("<Q", offset)
            for offset in offsets
        )
    )

    manifest = {
        "dataset": "dictionary-full-v1",
        "granularity": "one-passage-per-sense-fragment",
        "max_chars": MAX_CHARS,
        "model": MODEL_ID,
        "source_dim": 768,
        "matryoshka": True,
        "doc_prompt": "title: {title} | text: {text}",
        "query_prompt": (
            "task: search result | query: {query}"
        ),
        "dim": DIM,
        "count": total_passages,
        "quant": "int8",
        "scale": 127,
        "meta_codec": "zstd-blocks",
        "meta_rows_per_block": ROWS_PER_BLOCK,
        "url_template": (
            "https://en.wiktionary.org/wiki/{title}"
        ),
        "attribution": (
            "Wiktionary contributors; "
            "extracted with Wiktextract"
        ),
        "license": (
            "CC BY-SA 4.0; retain source attribution "
            "and share-alike terms"
        ),
        "model_sha256": MODEL_SHA256,
        "source_english_entries": source_count,
        "source_records_with_passages": sum(
            report["entries_with_passages"]
            for report in reports
        ),
        "source_records_without_passages": sum(
            report["entries_without_passages"]
            for report in reports
        ),
        "duplicate_headwords_preserved": True,
        "all_usable_senses_preserved": True,
    }

    manifest_path = output / "manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("Validating every metadata block...", flush=True)

    decompressor = zstd.ZstdDecompressor()

    with metadata_path.open("rb") as source:
        for index, start in enumerate(offsets[:-1]):
            source.seek(start)

            compressed = source.read(
                offsets[index + 1] - start
            )

            lines = decompressor.decompress(
                compressed
            ).decode("utf-8").splitlines()

            expected_rows = min(
                ROWS_PER_BLOCK,
                total_passages - index * ROWS_PER_BLOCK,
            )

            if len(lines) != expected_rows:
                raise SystemExit(
                    f"Invalid metadata block {index}"
                )

            for line in lines:
                row = json.loads(line)

                if not row.get("title") or not row.get("text"):
                    raise SystemExit(
                        f"Invalid row in metadata block {index}"
                    )

                if len(row["text"]) > MAX_CHARS:
                    raise SystemExit(
                        "Metadata passage exceeds character limit"
                    )

    report = {
        "validated": True,
        "revision": manifest["dataset"],
        "source_english_entries": source_count,
        "source_records_with_passages": (
            manifest["source_records_with_passages"]
        ),
        "source_records_without_passages": (
            manifest["source_records_without_passages"]
        ),
        "passage_count": total_passages,
        "dimension": DIM,
        "vector_bytes": vectors_path.stat().st_size,
        "metadata_bytes": metadata_path.stat().st_size,
        "offset_bytes": offsets_path.stat().st_size,
        "metadata_blocks": len(offsets) - 1,
        "metadata_rows_validated": rows_seen,
        "duplicate_headwords_preserved": True,
        "files": {
            filename: {
                "bytes": (output / filename).stat().st_size,
                "sha256": sha256_file(output / filename),
            }
            for filename in (
                "manifest.json",
                "vectors.i8",
                "meta.zst",
                "meta.offsets",
            )
        },
    }

    (output / "pack-report.json").write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
