
#!/usr/bin/env python3
"""Validate the full Ensu dictionary pack and test it with Ensu's Rust reader."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct

import zstandard as zstd

IDENTITY = "dictionary-full-v1"
COUNT = 2_008_148
DIM = 512
ROWS_PER_BLOCK = 128
MAX_CHARS = 600
URL_TEMPLATE = "https://en.wiktionary.org/wiki/{title}"

EXPECTED_HASHES = {
    "manifest.json": "74184b180cc32dacd0c7cb93ccd42bf5b265eff6f70f6f59d9b28bf503b84fcd",
    "vectors.i8": "761732f3cf01559b113372e5ec9a1ab45f879b843854aa2fef9407e3ed033039",
    "meta.zst": "58c3d2bc663e91661648c05f468c32f6763548b2e38ba66e83b5f86ed6f5f601",
    "meta.offsets": "ca3be05588102bb6defff6cc64723ca7d48628b469fd793a78103b9c6da85e24",
}

REQUIRED = tuple(EXPECTED_HASHES)

RUST_TEST = r'''
    #[test]
    fn opens_full_dictionary_pack_with_real_retrieval_index() {
        use std::io::Read;

        let pack_dir = std::path::PathBuf::from(
            std::env::var("ENSU_DICTIONARY_FULL_DIR")
                .expect("full dictionary directory env"),
        );

        // Reuse an existing upstream dataset configuration.
        // The full pack identity and URL are overridden below.
        let mut expected = crate::config::knowledge_dataset("simplewiki")
            .expect("existing Ensu dataset config");

        expected.current_download_identity = "dictionary-full-v1".to_owned();
        expected.max_chars = 600;
        expected.source_url_template =
            "https://en.wiktionary.org/wiki/{title}".to_owned();

        let index = RetrievalIndex::open(&pack_dir, &expected).unwrap();

        assert_eq!(index.dataset_identity(), "dictionary-full-v1");
        assert_eq!(index.count, 2_008_148);
        assert_eq!(index.dim, 512);

        // Read only one real vector instead of loading the 1 GB file.
        let mut vector_file =
            std::fs::File::open(pack_dir.join(KNOWLEDGE_VECTORS_FILE)).unwrap();

        let mut query_bytes = vec![0_u8; index.dim];
        vector_file.read_exact(&mut query_bytes).unwrap();

        let query = query_bytes
            .iter()
            .map(|value| f32::from(*value as i8) / 127.0)
            .collect::<Vec<_>>();

        // Search the complete vector file through Ensu's actual Rust reader.
        let hits = index.search(&query, 10, 0.0).unwrap();

        assert!(!hits.is_empty());
        assert!(hits.iter().all(|hit| {
            !hit.title.trim().is_empty()
                && !hit.text.trim().is_empty()
                && hit.source_url.starts_with(
                    "https://en.wiktionary.org/wiki/"
                )
        }));

        // Test passage reloading using a real Ensu locator.
        let first = &hits[0];
        let reloaded = index
            .reload_passage(&first.locator)
            .unwrap()
            .unwrap();

        assert_eq!(reloaded.title, first.title);
        assert_eq!(reloaded.text, first.text);
        assert_eq!(reloaded.source_url, first.source_url);
    }
'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_pack(pack: Path) -> dict:
    for name in REQUIRED:
        path = pack / name

        if not path.is_file():
            raise SystemExit(f"Missing pack file: {path}")

        actual = sha256_file(path)

        if actual != EXPECTED_HASHES[name]:
            raise SystemExit(
                f"SHA-256 mismatch for {name}: {actual}"
            )

        print(f"Verified SHA-256: {name}", flush=True)

    manifest = json.loads(
        (pack / "manifest.json").read_text(encoding="utf-8")
    )

    expected_manifest = {
        "dataset": IDENTITY,
        "count": COUNT,
        "dim": DIM,
        "max_chars": MAX_CHARS,
        "meta_rows_per_block": ROWS_PER_BLOCK,
        "url_template": URL_TEMPLATE,
    }

    for key, expected in expected_manifest.items():
        actual = manifest.get(key)

        if actual != expected:
            raise SystemExit(
                f"Manifest {key} mismatch: "
                f"{actual!r} != {expected!r}"
            )

    vector_size = (pack / "vectors.i8").stat().st_size

    if vector_size != COUNT * DIM:
        raise SystemExit(
            f"Vector size mismatch: {vector_size} != {COUNT * DIM}"
        )

    offset_bytes = (pack / "meta.offsets").read_bytes()
    block_count = (COUNT + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK

    if len(offset_bytes) != (block_count + 1) * 8:
        raise SystemExit("Metadata offset file has an unexpected size")

    offsets = struct.unpack(
        f"<{block_count + 1}Q",
        offset_bytes,
    )

    meta_size = (pack / "meta.zst").stat().st_size

    if offsets[0] != 0 or offsets[-1] != meta_size:
        raise SystemExit(
            "Metadata offsets do not cover the compressed metadata file"
        )

    if any(a >= b for a, b in zip(offsets, offsets[1:])):
        raise SystemExit("Metadata offsets are not strictly increasing")

    decompressor = zstd.ZstdDecompressor()
    total_rows = 0

    with (pack / "meta.zst").open("rb") as meta:
        for block in range(block_count):
            start = offsets[block]
            end = offsets[block + 1]

            meta.seek(start)
            compressed = meta.read(end - start)

            if len(compressed) != end - start:
                raise SystemExit(
                    f"Could not read metadata block {block}"
                )

            decoded = decompressor.decompress(compressed)
            lines = decoded.splitlines()

            expected_rows = min(
                ROWS_PER_BLOCK,
                COUNT - total_rows,
            )

            if len(lines) != expected_rows:
                raise SystemExit(
                    f"Block {block} has {len(lines)} rows; "
                    f"expected {expected_rows}"
                )

            for line in lines:
                row = json.loads(line)

                if (
                    not isinstance(row.get("title"), str)
                    or not row["title"].strip()
                ):
                    raise SystemExit(
                        f"Block {block} contains an empty title"
                    )

                if (
                    not isinstance(row.get("text"), str)
                    or not row["text"].strip()
                ):
                    raise SystemExit(
                        f"Block {block} contains empty passage text"
                    )

                if len(row["text"]) > MAX_CHARS:
                    raise SystemExit(
                        f"Block {block} contains text longer "
                        f"than {MAX_CHARS} characters"
                    )

            total_rows += len(lines)

            if block % 2000 == 0:
                print(
                    f"Validated metadata blocks: "
                    f"{block}/{block_count}",
                    flush=True,
                )

    if total_rows != COUNT:
        raise SystemExit(
            f"Metadata row count mismatch: {total_rows} != {COUNT}"
        )

    return {
        "dataset": IDENTITY,
        "passages": total_rows,
        "dimensions": DIM,
        "metadata_blocks": block_count,
        "vector_bytes": vector_size,
        "sha256": EXPECTED_HASHES,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pack-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--upstream-root",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    pack = args.pack_dir.resolve()
    report = validate_pack(pack)

    print(
        "FULL_PACK_STRUCTURAL_REPORT "
        + json.dumps(report),
        flush=True,
    )

    upstream = args.upstream_root.resolve()

    index_rs = (
        upstream
        / "rust/crates/ensu/src/retrieval/index.rs"
    )

    if not index_rs.is_file():
        raise SystemExit("Ensu Rust index source not found")

    if not (upstream / "rust/Cargo.toml").is_file():
        raise SystemExit("Ensu Rust Cargo workspace not found")

    marker = "#[cfg(test)]\npub(super) mod tests {"

    original = index_rs.read_text(encoding="utf-8")

    test_name = (
        "opens_full_dictionary_pack_with_real_retrieval_index"
    )

    if marker not in original:
        raise SystemExit(
            "Could not find Ensu RetrievalIndex test module"
        )

    if test_name in original:
        raise SystemExit(
            "Full-pack Rust test already exists in upstream source"
        )

    patched = original.replace(
        marker,
        marker + "\n" + RUST_TEST,
        1,
    )

    index_rs.write_text(patched, encoding="utf-8")

    env = os.environ.copy()
    env["ENSU_DICTIONARY_FULL_DIR"] = str(pack)
    env.setdefault("CARGO_TERM_COLOR", "never")

    command = [
        "cargo",
        "test",
        "-p",
        "ente-ensu",
        "--lib",
        "--no-default-features",
        f"retrieval::index::tests::{test_name}",
        "--",
        "--nocapture",
    ]

    print(
        "Running full-pack test through Ensu's real Rust RetrievalIndex...",
        flush=True,
    )

    subprocess.run(
        command,
        cwd=upstream / "rust",
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
