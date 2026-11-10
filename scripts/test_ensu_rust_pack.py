#!/usr/bin/env python3
'Run the real Ensu Rust RetrievalIndex against a supplied pilot pack.'

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
    fn opens_dictionary_pilot_pack_with_real_retrieval_index() {
        let pack_dir = std::path::PathBuf::from(
            std::env::var("ENSU_DICTIONARY_PILOT_DIR").expect("pilot directory env"),
        );
        let expected = crate::config::knowledge_dataset("simplewiki")
            .expect("Ensu simplewiki dataset config");
        let index = RetrievalIndex::open(&pack_dir, &expected).unwrap();

        assert_eq!(index.dataset_identity(), "dictionary-pilot-v1");
        assert_eq!(index.count, 32);
        assert_eq!(index.dim, 512);

        // A zero query scores every row at 0.0. This forces the real Rust
        // reader to decompress and parse every metadata row in the pilot.
        let zero_query = vec![0.0_f32; index.dim];
        let all_hits = index.search(&zero_query, 32, 0.0).unwrap();
        assert_eq!(all_hits.len(), 32);
        assert!(all_hits.iter().all(|hit| {
            !hit.title.is_empty()
                && !hit.text.trim().is_empty()
                && hit.source_url.starts_with("https://simple.wikipedia.org/wiki/")
        }));

        // Reload one selected passage through its real Ensu locator.
        let first = &all_hits[0];
        let reloaded = index.reload_passage(&first.locator).unwrap().unwrap();
        assert_eq!(reloaded.title, first.title);
        assert_eq!(reloaded.text, first.text);
        assert_eq!(reloaded.source_url, first.source_url);

        // Also exercise the int8 vector search path using a real pilot vector.
        let vector_bytes = std::fs::read(pack_dir.join(KNOWLEDGE_VECTORS_FILE)).unwrap();
        let query = vector_bytes[..index.dim]
            .iter()
            .map(|value| f32::from(*value as i8) / 127.0)
            .collect::<Vec<_>>();
        let ranked = index.search(&query, 5, 0.0).unwrap();
        assert!(!ranked.is_empty());
    }
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--pilot-zip", required=True, type=Path)
    args = parser.parse_args()

    upstream = args.upstream_root.resolve()
    zip_path = args.pilot_zip.resolve()
    index_rs = upstream / "rust/crates/ensu/src/retrieval/index.rs"
    cargo_toml = upstream / "rust/Cargo.toml"
    if not index_rs.is_file() or not cargo_toml.is_file():
        raise SystemExit("Could not find the Ensu Rust source tree under --upstream-root.")
    if not zip_path.is_file():
        raise SystemExit(f"Pilot ZIP not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        required = {"manifest.json", "vectors.i8", "meta.zst", "meta.offsets"}
        if not required.issubset(names):
            raise SystemExit(f"Pilot ZIP is missing files: {sorted(required - names)}")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("dataset") != "dictionary-pilot-v1":
            raise SystemExit("Unexpected manifest dataset; expected dictionary-pilot-v1.")
        if manifest.get("url_template") != "https://simple.wikipedia.org/wiki/{title}":
            raise SystemExit("Pilot URL template does not match Ensu's simplewiki test config.")
        if manifest.get("count") != 32 or manifest.get("dim") != 512:
            raise SystemExit("Expected the 32-row, 512-dimension pilot pack.")

        pilot_dir = upstream / "target/ensu-dictionary-pilot/dictionary-pilot-v1"
        if pilot_dir.exists():
            shutil.rmtree(pilot_dir)
        pilot_dir.mkdir(parents=True)
        for filename in required:
            (pilot_dir / filename).write_bytes(archive.read(filename))

    original = index_rs.read_text(encoding="utf-8")
    marker = "#[cfg(test)]\npub(super) mod tests {"
    if marker not in original:
        raise SystemExit("Could not locate the Ensu RetrievalIndex unit-test module; source changed.")
    if "opens_dictionary_pilot_pack_with_real_retrieval_index" in original:
        raise SystemExit("The pilot Rust test is already present in the checked-out source.")
    patched = original.replace(marker, marker + "\n" + TEST, 1)
    index_rs.write_text(patched, encoding="utf-8")

    env = os.environ.copy()
    env["ENSU_DICTIONARY_PILOT_DIR"] = str(pilot_dir)
    env.setdefault("CARGO_TERM_COLOR", "never")
    command = [
        "cargo", "test", "-p", "ente-ensu", "--lib", "--no-default-features",
        "retrieval::index::tests::opens_dictionary_pilot_pack_with_real_retrieval_index",
        "--", "--nocapture",
    ]
    print("Testing the uploaded pilot with Ensu's actual Rust RetrievalIndex...")
    print(f"Pilot folder: {pilot_dir}")
    subprocess.run(command, cwd=upstream / "rust", env=env, check=True)

if __name__ == "__main__":
    main()
