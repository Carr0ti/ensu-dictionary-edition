
#!/usr/bin/env python3
"""Test the 100K dictionary pack with Ensu's real Rust RetrievalIndex."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


TEST = r'''
    #[test]
    fn opens_and_reads_dictionary_100k_pack_with_real_retrieval_index() {
        let pack_dir = std::path::PathBuf::from(
            std::env::var("ENSU_DICTIONARY_100K_DIR")
                .expect("100K directory env"),
        );

        let expected = crate::config::knowledge_dataset("ensu-dictionary")
            .expect("Ensu dictionary dataset config");

        assert_eq!(expected.stable_id, "ensu-dictionary");
        assert_eq!(
            expected.current_download_identity,
            "dictionary-100k-v1"
        );
        assert_eq!(
            expected.source_url_template,
            "https://en.wiktionary.org/wiki/{title}"
        );
        assert!(expected.attribution.credit.contains("Wiktionary"));
        assert_eq!(
            expected.attribution.license_label,
            "CC BY-SA 4.0"
        );
        assert_eq!(
            expected.attribution.public_pack_url,
            "https://github.com/Carr0ti/ensu-dictionary-edition/releases/tag/dictionary-100k-v1"
        );

        let index = RetrievalIndex::open(&pack_dir, &expected).unwrap();

        assert_eq!(index.dataset_identity(), "dictionary-100k-v1");
        assert_eq!(index.count, 100_000);
        assert_eq!(index.dim, 512);

        // Read all 100,000 passages through Ensu's real reader.
        let zero_query = vec![0.0_f32; index.dim];
        let all_hits = index.search(&zero_query, 100_000, 0.0).unwrap();

        assert_eq!(all_hits.len(), 100_000);

        assert!(all_hits.iter().all(|hit| {
            !hit.title.trim().is_empty()
                && !hit.text.trim().is_empty()
                && hit.source_url.starts_with(
                    "https://en.wiktionary.org/wiki/"
                )
        }));

        // Test Ensu's real passage reload mechanism.
        let first = &all_hits[0];
        let reloaded = index.reload_passage(&first.locator).unwrap().unwrap();

        assert_eq!(reloaded.title, first.title);
        assert_eq!(reloaded.text, first.text);
        assert_eq!(reloaded.source_url, first.source_url);

        // Exercise the int8 vector scoring path.
        let vector_bytes =
            std::fs::read(pack_dir.join(KNOWLEDGE_VECTORS_FILE)).unwrap();

        let query = vector_bytes[..index.dim]
            .iter()
            .map(|value| f32::from(*value as i8) / 127.0)
            .collect::<Vec<_>>();

        let ranked = index.search(&query, 10, 0.0).unwrap();
        assert!(!ranked.is_empty());
    }
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream-root", required=True, type=Path)
    ap.add_argument("--pack-dir", required=True, type=Path)
    args = ap.parse_args()

    upstream = args.upstream_root.resolve()
    source_pack = args.pack_dir.resolve()

    index_rs = upstream / "rust/crates/ensu/src/retrieval/index.rs"

    if not index_rs.is_file() or not (upstream / "rust/Cargo.toml").is_file():
        raise SystemExit("Could not find Ensu Rust source tree.")

    required = {
        "manifest.json",
        "vectors.i8",
        "meta.zst",
        "meta.offsets",
    }

    if not source_pack.is_dir():
        raise SystemExit(f"Pack directory not found: {source_pack}")

    report_path = source_pack / "pack-report.json"
    manifest_path = source_pack / "manifest.json"

    if not report_path.is_file() or not manifest_path.is_file():
        raise SystemExit("Pack manifest or validation report is missing.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    if manifest.get("dataset") != "dictionary-100k-v1":
        raise SystemExit("Unexpected dataset identity.")

    if manifest.get("count") != 100_000 or manifest.get("dim") != 512:
        raise SystemExit("Expected 100,000 passages and 512 dimensions.")

    if not report.get("validated"):
        raise SystemExit("Pack builder validation did not pass.")

    if report.get("metadata_rows_validated") != 100_000:
        raise SystemExit("Expected 100,000 validated metadata rows.")

    # Verify every pack file against the builder's report.
    for filename in required:
        path = source_pack / filename

        if not path.is_file():
            raise SystemExit(f"Missing pack file: {filename}")

        expected_file = report["files"][filename]
        data = path.read_bytes()

        if len(data) != expected_file["bytes"]:
            raise SystemExit(f"File size mismatch: {filename}")

        digest = hashlib.sha256(data).hexdigest()

        if digest != expected_file["sha256"]:
            raise SystemExit(f"SHA256 mismatch: {filename}")

    # Copy the pack into Ensu's expected dataset directory.
    pack_dir = (
        upstream
        / "target/ensu-dictionary-100k/dictionary-100k-v1"
    )

    if pack_dir.exists():
        shutil.rmtree(pack_dir)

    pack_dir.mkdir(parents=True)

    for filename in required:
        shutil.copy2(source_pack / filename, pack_dir / filename)

    # IMPORTANT:
    # Apply this fork's dictionary catalog to the upstream Ensu source.
    project_root = Path(__file__).resolve().parents[1]

    dictionary_config = (
        project_root / "rust/crates/ensu/src/config/knowledge.rs"
    )

    upstream_config = (
        upstream / "rust/crates/ensu/src/config/knowledge.rs"
    )

    if not dictionary_config.is_file():
        raise SystemExit(
            f"Dictionary config not found: {dictionary_config}"
        )

    shutil.copy2(dictionary_config, upstream_config)

    print("Applied Ensu Dictionary Edition dataset catalog.")

    # Inject the integration test into Ensu's real RetrievalIndex tests.
    original = index_rs.read_text(encoding="utf-8")

    marker = "#[cfg(test)]\npub(super) mod tests {"

    if marker not in original:
        raise SystemExit(
            "Could not locate Ensu RetrievalIndex test module."
        )

    test_name = (
        "opens_and_reads_dictionary_100k_pack_with_real_retrieval_index"
    )

    if test_name in original:
        raise SystemExit("The 100K Rust test is already present.")

    index_rs.write_text(
        original.replace(marker, marker + "\n" + TEST, 1),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["ENSU_DICTIONARY_100K_DIR"] = str(pack_dir)
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

    print("Testing all 100,000 passages with Ensu's real Rust reader...")
    print(f"Pack folder: {pack_dir}")

    subprocess.run(
        command,
        cwd=upstream / "rust",
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
