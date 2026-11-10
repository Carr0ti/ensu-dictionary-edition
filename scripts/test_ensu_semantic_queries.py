#!/usr/bin/env python3
"""Insert a temporary test into Ensu and run real Rust semantic retrieval."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, zipfile
from pathlib import Path

TEST = r'''
    #[test]
    fn runs_meaning_based_queries_through_real_retrieval_index() {
        use std::fs;
        use serde_json::{json, Value};

        let pack_dir = std::path::PathBuf::from(
            std::env::var("ENSU_DICTIONARY_PILOT_DIR").expect("pilot directory env"),
        );
        let fixture_path = std::env::var("ENSU_SEMANTIC_QUERY_FIXTURES")
            .expect("semantic query fixture env");
        let report_path = std::env::var("ENSU_SEMANTIC_RUST_RESULTS")
            .expect("semantic Rust report env");
        let expected = crate::config::knowledge_dataset("simplewiki")
            .expect("Ensu simplewiki dataset config");
        let index = RetrievalIndex::open(&pack_dir, &expected).unwrap();
        assert_eq!(index.dataset_identity(), "dictionary-pilot-v1");

        let fixtures: Vec<Value> =
            serde_json::from_slice(&fs::read(fixture_path).unwrap()).unwrap();
        assert!(fixtures.len() >= 5, "expected several meaning-based queries");

        let mut results = Vec::new();
        for fixture in fixtures {
            let query = fixture["query"].as_str().expect("query text").to_owned();
            let expected_headword = fixture["expected_headword"]
                .as_str().expect("expected headword").to_owned();
            let vector = fixture["vector"].as_array().expect("query vector")
                .iter()
                .map(|value| value.as_f64().expect("numeric vector component") as f32)
                .collect::<Vec<_>>();
            assert_eq!(vector.len(), 512, "query vector must have 512 dimensions");

            let hits = index.search(&vector, 10, 0.0).unwrap();
            assert!(!hits.is_empty(), "query returned no hits: {query}");
            let expected_rank = hits.iter()
                .position(|hit| hit.title.eq_ignore_ascii_case(&expected_headword))
                .map(|rank| rank + 1);
            let top_results = hits.iter().map(|hit| {
                json!({"title": &hit.title, "score": hit.score})
            }).collect::<Vec<_>>();

            println!(
                "SEMANTIC_RESULT {}",
                json!({
                    "query": query,
                    "expected_headword": expected_headword,
                    "expected_rank": expected_rank,
                    "top_results": top_results
                })
            );
            results.push(json!({
                "query": query,
                "expected_headword": expected_headword,
                "expected_rank": expected_rank,
                "top_results": top_results
            }));
        }

        fs::write(report_path, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
        println!("Meaning-based Rust retrieval completed for {} queries.", results.len());
    }
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream-root", type=Path, required=True)
    ap.add_argument("--pilot-zip", type=Path, required=True)
    ap.add_argument("--query-fixtures", type=Path, required=True)
    ap.add_argument("--rust-results", type=Path, required=True)
    args = ap.parse_args()

    upstream = args.upstream_root.resolve()
    index_rs = upstream / "rust/crates/ensu/src/retrieval/index.rs"
    if not index_rs.is_file() or not (upstream / "rust/Cargo.toml").is_file():
        raise SystemExit("Ensu Rust source tree not found")
    if not args.pilot_zip.is_file() or not args.query_fixtures.is_file():
        raise SystemExit("Pilot ZIP or query fixtures are missing")

    with zipfile.ZipFile(args.pilot_zip) as archive:
        names = set(archive.namelist())
        required = {"manifest.json", "vectors.i8", "meta.zst", "meta.offsets"}
        if not required.issubset(names):
            raise SystemExit(f"Pilot ZIP missing files: {sorted(required - names)}")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("dataset") != "dictionary-pilot-v1":
            raise SystemExit("Unexpected pilot dataset identity")
        if manifest.get("url_template") != "https://simple.wikipedia.org/wiki/{title}":
            raise SystemExit("Pilot URL template does not match Ensu test config")
        if manifest.get("count") != 32 or manifest.get("dim") != 512:
            raise SystemExit("Expected the existing 32-passage, 512-dimension pilot")

        pilot_dir = upstream / "target/ensu-dictionary-pilot/dictionary-pilot-v1"
        if pilot_dir.exists():
            shutil.rmtree(pilot_dir)
        pilot_dir.mkdir(parents=True)
        for filename in required:
            (pilot_dir / filename).write_bytes(archive.read(filename))

    original = index_rs.read_text(encoding="utf-8")
    marker = "#[cfg(test)]\npub(super) mod tests {"
    if marker not in original:
        raise SystemExit("Could not find Ensu RetrievalIndex test module; upstream source may have changed")
    test_name = "runs_meaning_based_queries_through_real_retrieval_index"
    if test_name in original:
        raise SystemExit("Semantic retrieval test is already present")
    index_rs.write_text(original.replace(marker, marker + "\n" + TEST, 1), encoding="utf-8")

    args.rust_results.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["ENSU_DICTIONARY_PILOT_DIR"] = str(pilot_dir)
    env["ENSU_SEMANTIC_QUERY_FIXTURES"] = str(args.query_fixtures.resolve())
    env["ENSU_SEMANTIC_RUST_RESULTS"] = str(args.rust_results.resolve())
    env.setdefault("CARGO_TERM_COLOR", "never")

    command = [
        "cargo", "test", "-p", "ente-ensu", "--lib", "--no-default-features",
        f"retrieval::index::tests::{test_name}", "--", "--nocapture",
    ]
    print("Running meaning-based queries through Ensu's real Rust RetrievalIndex...")
    subprocess.run(command, cwd=upstream / "rust", env=env, check=True)

if __name__ == "__main__":
    main()
