
#!/usr/bin/env python3
"""Repair the 100K pack's source URL without rebuilding embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import zstandard as zstd


URL_TEMPLATE = "https://en.wiktionary.org/wiki/{title}"
EXPECTED_DATASET = "dictionary-100k-v1"
EXPECTED_COUNT = 100_000
EXPECTED_DIM = 512
ROWS_PER_BLOCK = 128

FILES = (
    "manifest.json",
    "vectors.i8",
    "meta.zst",
    "meta.offsets",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def repair(pack_dir: Path) -> None:
    manifest_path = pack_dir / "manifest.json"
    report_path = pack_dir / "pack-report.json"

    if not manifest_path.is_file() or not report_path.is_file():
        raise SystemExit("Missing manifest.json or pack-report.json.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    if manifest.get("dataset") != EXPECTED_DATASET:
        raise SystemExit("Unexpected dictionary pack identity.")

    if manifest.get("count") != EXPECTED_COUNT:
        raise SystemExit("Expected exactly 100,000 passages.")

    if manifest.get("dim") != EXPECTED_DIM:
        raise SystemExit("Expected 512 embedding dimensions.")

    if not report.get("validated"):
        raise SystemExit("Original builder validation did not pass.")

    if report.get("metadata_rows_validated") != EXPECTED_COUNT:
        raise SystemExit("Original report does not validate 100,000 rows.")

    for filename in FILES:
        if not (pack_dir / filename).is_file():
            raise SystemExit(f"Missing pack file: {filename}")

    # Correct the stale Simple Wikipedia URL.
    manifest["url_template"] = URL_TEMPLATE

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Validate vector dimensions.
    vector_size = (pack_dir / "vectors.i8").stat().st_size

    if vector_size != EXPECTED_COUNT * EXPECTED_DIM:
        raise SystemExit("Vector file size is incorrect.")

    # Validate metadata offsets.
    metadata_path = pack_dir / "meta.zst"
    offsets_path = pack_dir / "meta.offsets"

    metadata_size = metadata_path.stat().st_size
    raw_offsets = offsets_path.read_bytes()

    block_count = (
        EXPECTED_COUNT + ROWS_PER_BLOCK - 1
    ) // ROWS_PER_BLOCK

    expected_offset_bytes = (block_count + 1) * 8

    if len(raw_offsets) != expected_offset_bytes:
        raise SystemExit("Metadata offset file has an incorrect size.")

    offsets = struct.unpack(
        "<" + "Q" * (block_count + 1),
        raw_offsets,
    )

    if offsets[0] != 0 or offsets[-1] != metadata_size:
        raise SystemExit("Metadata offsets do not match metadata file.")

    if any(a >= b for a, b in zip(offsets, offsets[1:])):
        raise SystemExit("Metadata offsets are not strictly increasing.")

    # Decompress and validate every metadata block.
    decompressor = zstd.ZstdDecompressor()
    rows_seen = 0
    maximum_chars = 0

    with metadata_path.open("rb") as metadata:
        for block in range(block_count):
            metadata.seek(offsets[block])
            compressed = metadata.read(
                offsets[block + 1] - offsets[block]
            )

            decoded = decompressor.decompress(compressed)
            lines = decoded.decode("utf-8").splitlines()

            expected_rows = min(
                ROWS_PER_BLOCK,
                EXPECTED_COUNT - block * ROWS_PER_BLOCK,
            )

            if len(lines) != expected_rows:
                raise SystemExit(
                    f"Block {block} has {len(lines)} rows; "
                    f"expected {expected_rows}."
                )

            for line in lines:
                row = json.loads(line)

                title = str(row.get("title", "")).strip()
                text = str(row.get("text", "")).strip()

                if not title or not text:
                    raise SystemExit(
                        f"Invalid metadata row in block {block}."
                    )

                if len(text) > 600:
                    raise SystemExit(
                        f"Passage exceeds 600 characters: {title}"
                    )

                rows_seen += 1
                maximum_chars = max(maximum_chars, len(text))

    if rows_seen != EXPECTED_COUNT:
        raise SystemExit("Metadata row count is incorrect.")

    # Recalculate hashes and sizes after manifest correction.
    report["validated"] = True
    report["revision"] = EXPECTED_DATASET
    report["passage_count"] = EXPECTED_COUNT
    report["unique_headwords"] = EXPECTED_COUNT
    report["dimension"] = EXPECTED_DIM
    report["vector_bytes"] = vector_size
    report["metadata_bytes"] = metadata_size
    report["metadata_blocks"] = block_count
    report["metadata_rows_validated"] = rows_seen
    report["max_passage_chars"] = maximum_chars

    report["files"] = {
        filename: {
            "bytes": (pack_dir / filename).stat().st_size,
            "sha256": sha256_file(pack_dir / filename),
        }
        for filename in FILES
    }

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("100K dictionary pack repair successful.")
    print(f"Source URL: {URL_TEMPLATE}")
    print(f"Metadata rows checked: {rows_seen:,}")
    print(f"Metadata blocks checked: {block_count:,}")
    print("All four pack-file hashes recalculated.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", required=True, type=Path)
    args = parser.parse_args()

    pack_dir = args.pack_dir.resolve()

    if not pack_dir.is_dir():
        raise SystemExit(f"Pack directory not found: {pack_dir}")

    repair(pack_dir)


if __name__ == "__main__":
    main()
