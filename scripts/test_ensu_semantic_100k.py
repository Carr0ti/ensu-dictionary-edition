
#!/usr/bin/env python3
"""Diagnose semantic ranking in the real Ensu 100K dictionary index."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess

import zstandard as zstd

MODEL_SHA256 = (
    "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63"
)

QUERIES = [
    ("mitigate", "to make a problem less serious or severe"),
    ("astronomer", "a person who studies stars and planets"),
    ("nevertheless", "despite what has just been said; even so"),
    ("deproblematize", "to reduce the severity or seriousness of a problem"),
    ("thermometer", "a device used to measure temperature"),
    ("physician", "a person who treats diseases"),
    ("ocean", "a very large body of salt water"),
    ("bicycle", "a vehicle with two wheels powered by pedals"),
]

RUST_TEST = r'''
    #[test]
    fn diagnoses_semantic_queries_against_dictionary_100k() {
        use serde_json::{json, Value};
        use std::fs;

        let pack_dir = std::path::PathBuf::from(
            std::env::var("ENSU_SEMANTIC_100K_DIR").unwrap(),
        );

        let fixture_path =
            std::env::var("ENSU_SEMANTIC_100K_FIXTURES").unwrap();

        let output_path =
            std::env::var("ENSU_SEMANTIC_100K_RESULTS").unwrap();

        let expected = crate::config::knowledge_dataset("ensu-dictionary")
            .expect("dictionary dataset config");

        let index = RetrievalIndex::open(&pack_dir, &expected).unwrap();

        assert_eq!(index.dataset_identity(), "dictionary-100k-v1");
        assert_eq!(index.count, 100_000);
        assert_eq!(index.dim, 512);

        let fixtures: Vec<Value> =
            serde_json::from_slice(&fs::read(fixture_path).unwrap()).unwrap();

        let mut output = Vec::new();

        for fixture in fixtures {
            let query = fixture["query"].as_str().unwrap();
            let expected_word = fixture["expected_word"].as_str().unwrap();

            let vector = fixture["vector"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect::<Vec<_>>();

            assert_eq!(vector.len(), 512);

            // Use threshold zero to inspect ranking before app filtering.
            let hits = index.search(&vector, 10, 0.0).unwrap();

            let top = hits.iter().map(|hit| {
                json!({
                    "title": hit.title,
                    "score": hit.score
                })
            }).collect::<Vec<_>>();

            let result = json!({
                "query": query,
                "expected_word": expected_word,
                "top_results": top
            });

            println!("SEMANTIC_DIAGNOSTIC {}", result);
            output.push(result);
        }

        fs::write(
            output_path,
            serde_json::to_vec_pretty(&output).unwrap(),
        ).unwrap();

        println!("Completed {} diagnostic queries.", output.len());
    }
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_titles(pack: Path) -> set[str]:
    compressed = (pack / "meta.zst").read_bytes()
    raw_offsets = (pack / "meta.offsets").read_bytes()

    if len(raw_offsets) % 8:
        raise SystemExit("Invalid metadata offsets file")

    offsets = struct.unpack(
        "<" + "Q" * (len(raw_offsets) // 8),
        raw_offsets,
    )

    if len(offsets) < 2 or offsets[0] != 0:
        raise SystemExit("Invalid metadata block offsets")

    if offsets[-1] != len(compressed):
        raise SystemExit("Metadata offsets do not match compressed file")

    titles = set()
    decoder = zstd.ZstdDecompressor()

    for start, end in zip(offsets, offsets[1:]):
        block = decoder.decompress(compressed[start:end])

        for line in block.decode("utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                title = str(row.get("title", "")).strip()
                if title:
                    titles.add(title.casefold())

    return titles


def embed_queries(model_path: Path, output: Path) -> None:
    from llama_cpp import Llama, LLAMA_POOLING_TYPE_MEAN

    model = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_batch=2048,
        n_ubatch=2048,
        n_threads=4,
        n_threads_batch=4,
        embedding=True,
        pooling_type=LLAMA_POOLING_TYPE_MEAN,
        n_gpu_layers=0,
        verbose=False,
    )

    fixtures = []

    for number, (expected, query) in enumerate(QUERIES, 1):
        prompt = f"task: search result | query: {query}"

        try:
            result = model.embed(prompt, normalize=False)
        except TypeError:
            result = model.embed(prompt)

        vector = (
            result[0]
            if result and isinstance(result[0], (list, tuple))
            else result
        )

        vector = [float(value) for value in vector]

        if len(vector) != 768 or not all(map(math.isfinite, vector)):
            raise SystemExit(f"Invalid embedding for query: {query}")

        vector = vector[:512]
        norm = math.sqrt(math.fsum(value * value for value in vector))

        if not math.isfinite(norm) or norm <= 0:
            raise SystemExit(f"Invalid vector norm for query: {query}")

        vector = [value / norm for value in vector]

        fixtures.append({
            "query": query,
            "expected_word": expected,
            "vector": vector,
        })

        print(f"Embedded query {number}/{len(QUERIES)}: {expected}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(fixtures, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pack = args.pack_dir.resolve()
    model = args.model.resolve()
    upstream = args.upstream_root.resolve()
    output = args.output.resolve()

    required = (
        "manifest.json",
        "vectors.i8",
        "meta.zst",
        "meta.offsets",
        "pack-report.json",
    )

    for name in required:
        if not (pack / name).is_file():
            raise SystemExit(f"Missing pack file: {name}")

    manifest = json.loads(
        (pack / "manifest.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (pack / "pack-report.json").read_text(encoding="utf-8")
    )

    if manifest.get("dataset") != "dictionary-100k-v1":
        raise SystemExit("Unexpected pack identity")

    if manifest.get("count") != 100_000 or manifest.get("dim") != 512:
        raise SystemExit("Unexpected pack dimensions")

    if not report.get("validated"):
        raise SystemExit("Pack validation report did not pass")

    for name in required[:4]:
        path = pack / name
        expected = report["files"][name]

        if path.stat().st_size != expected["bytes"]:
            raise SystemExit(f"Size mismatch: {name}")

        if sha256(path) != expected["sha256"]:
            raise SystemExit(f"SHA-256 mismatch: {name}")

    print("Pack files verified.")

    titles = read_titles(pack)
    print(f"Unique headwords: {len(titles):,}")

    for word, _ in QUERIES:
        print(
            f"HEADWORD {word}: "
            f"{'PRESENT' if word.casefold() in titles else 'MISSING'}"
        )

    if not model.is_file():
        raise SystemExit("Embedding model is missing")

    if sha256(model) != MODEL_SHA256:
        raise SystemExit("Embedding model SHA-256 mismatch")

    fixture_path = output.parent / "semantic-100k-fixtures.json"
    embed_queries(model, fixture_path)

    index_rs = upstream / "rust/crates/ensu/src/retrieval/index.rs"
    upstream_config = (
        upstream / "rust/crates/ensu/src/config/knowledge.rs"
    )
    project_root = Path(__file__).resolve().parents[1]
    project_config = (
        project_root / "rust/crates/ensu/src/config/knowledge.rs"
    )

    if not index_rs.is_file() or not (upstream / "rust/Cargo.toml").is_file():
        raise SystemExit("Ensu Rust source tree not found")

    shutil.copy2(project_config, upstream_config)

    marker = "#[cfg(test)]\npub(super) mod tests {"
    source = index_rs.read_text(encoding="utf-8")

    if marker not in source:
        raise SystemExit("Could not find RetrievalIndex test module")

    test_name = "diagnoses_semantic_queries_against_dictionary_100k"

    if test_name in source:
        raise SystemExit("Diagnostic test already exists in source")

    index_rs.write_text(
        source.replace(marker, marker + "\n" + RUST_TEST, 1),
        encoding="utf-8",
    )

    rust_pack = (
        upstream / "target/ensu-semantic-100k/dictionary-100k-v1"
    )

    if rust_pack.exists():
        shutil.rmtree(rust_pack)

    rust_pack.mkdir(parents=True)

    for name in required[:4]:
        shutil.copy2(pack / name, rust_pack / name)

    results_path = output.parent / "semantic-100k-rust-results.json"

    env = os.environ.copy()
    env["ENSU_SEMANTIC_100K_DIR"] = str(rust_pack)
    env["ENSU_SEMANTIC_100K_FIXTURES"] = str(fixture_path)
    env["ENSU_SEMANTIC_100K_RESULTS"] = str(results_path)
    env.setdefault("CARGO_TERM_COLOR", "never")

    command = [
        "cargo", "test",
        "-p", "ente-ensu",
        "--lib",
        "--no-default-features",
        f"retrieval::index::tests::{test_name}",
        "--",
        "--nocapture",
    ]

    subprocess.run(
        command,
        cwd=upstream / "rust",
        env=env,
        check=True,
    )

    results = json.loads(results_path.read_text(encoding="utf-8"))

    report_output = {
        "dataset": "dictionary-100k-v1",
        "query_count": len(results),
        "queries": results,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report_output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report_output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
