#!/usr/bin/env python3
"""Run Ensu's real Rust RetrievalIndex against the 10,000-passage pack."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

TEST = r'''
    #[test]
    fn opens_and_reads_dictionary_10k_pack_with_real_retrieval_index() {
        let pack_dir = std::path::PathBuf::from(
            std::env::var("ENSU_DICTIONARY_10K_DIR").expect("10k directory env"),
        );
        let expected = crate::config::knowledge_dataset("simplewiki")
            .expect("Ensu simplewiki dataset config");
        let index = RetrievalIndex::open(&pack_dir, &expected).unwrap();

        assert_eq!(index.dataset_identity(), "dictionary-10k-v1");
        assert_eq!(index.count, 10_000);
        assert_eq!(index.dim, 512);

        // A zero vector makes every row score 0.0. Asking for all rows forces
        // the real Rust reader to visit/decompress/parse every metadata block.
        let zero_query = vec![0.0_f32; index.dim];
        let all_hits = index.search(&zero_query, 10_000, 0.0).unwrap();
        assert_eq!(all_hits.len(), 10_000);
        assert!(all_hits.iter().all(|hit| {
            !hit.title.trim().is_empty()
                && !hit.text.trim().is_empty()
                && hit.source_url.starts_with("https://simple.wikipedia.org/wiki/")
        }));

        // Reload a passage using Ensu's real locator.
        let first = &all_hits[0];
        let reloaded = index.reload_passage(&first.locator).unwrap().unwrap();
        assert_eq!(reloaded.title, first.title);
        assert_eq!(reloaded.text, first.text);
        assert_eq!(reloaded.source_url, first.source_url);

        // Exercise the int8 vector scoring path with the first stored vector.
        let vector_bytes = std::fs::read(pack_dir.join(KNOWLEDGE_VECTORS_FILE)).unwrap();
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
    ap.add_argument("--pack-zip", required=True, type=Path)
    args = ap.parse_args()

    upstream = args.upstream_root.resolve()
    zip_path = args.pack_zip.resolve()
    index_rs = upstream / "rust/crates/ensu/src/retrieval/index.rs"
    if not index_rs.is_file() or not (upstream / "rust/Cargo.toml").is_file():
        raise SystemExit("Could not find Ensu Rust source tree under --upstream-root.")
    if not zip_path.is_file():
        raise SystemExit(f"10K pack ZIP not found: {zip_path}")

    required = {"manifest.json", "vectors.i8", "meta.zst", "meta.offsets"}
    prefix = "semantic-pack/dictionary/dictionary-10k-v1/"
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        if required.issubset(names):
            read_name = lambda filename: archive.read(filename)
        elif all(prefix + filename in names for filename in required):
            read_name = lambda filename: archive.read(prefix + filename)
        else:
            raise SystemExit(
                "Pack ZIP is missing required files. Expected flat names or "
                f"{prefix}<filename>."
            )

        manifest = json.loads(read_name("manifest.json"))
        if manifest.get("dataset") != "dictionary-10k-v1":
            raise SystemExit("Unexpected pack dataset identity.")
        if manifest.get("url_template") != "https://simple.wikipedia.org/wiki/{title}":
            raise SystemExit("URL template does not match Ensu simplewiki test config.")
        if manifest.get("count") != 10_000 or manifest.get("dim") != 512:
            raise SystemExit("Expected exactly 10,000 passages and 512 dimensions.")

        pack_dir = upstream / "target/ensu-dictionary-10k/dictionary-10k-v1"
        if pack_dir.exists():
            shutil.rmtree(pack_dir)
        pack_dir.mkdir(parents=True)
        for filename in required:
            (pack_dir / filename).write_bytes(read_name(filename))

    original = index_rs.read_text(encoding="utf-8")
    marker = "#[cfg(test)]\npub(super) mod tests {"
    if marker not in original:
        raise SystemExit("Could not locate Ensu RetrievalIndex test module; upstream source may have changed.")
    test_name = "opens_and_reads_dictionary_10k_pack_with_real_retrieval_index"
    if test_name in original:
        raise SystemExit("The 10K Rust test is already present in this checkout.")
    index_rs.write_text(original.replace(marker, marker + "\n" + TEST, 1), encoding="utf-8")

    env = os.environ.copy()
    env["ENSU_DICTIONARY_10K_DIR"] = str(pack_dir)
    env.setdefault("CARGO_TERM_COLOR", "never")
    command = [
        "cargo", "test", "-p", "ente-ensu", "--lib", "--no-default-features",
        f"retrieval::index::tests::{test_name}", "--", "--nocapture",
    ]
    print("Testing all 10,000 passages with Ensu's actual Rust RetrievalIndex...")
    print(f"Pack folder: {pack_dir}")
    subprocess.run(command, cwd=upstream / "rust", env=env, check=True)

if __name__ == "__main__":
    main()
