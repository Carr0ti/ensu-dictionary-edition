#!/usr/bin/env python3
"""Benchmark natural-language retrieval against the complete Ensu dictionary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

MODEL_URL = (
    "https://huggingface.co/ente-ai/embeddinggemma-300m-gguf/"
    "resolve/957b55764bd672f51240ac026e3a23ac9459ee3c/"
    "embeddinggemma-300M-Q8_0.gguf"
)
MODEL_SHA256 = (
    "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63"
)
MODEL_IDENTITY = (
    "ggml-org/embeddinggemma-300M-GGUF:"
    "embeddinggemma-300M-Q8_0.gguf"
)
MODEL_PATH = Path("build/models/embeddinggemma-300M-Q8_0.gguf")

QUERY_PROMPT = "task: search result | query: {query}"
DIM = 512
SOURCE_DIM = 768
IDENTITY = "dictionary-full-v1"
COUNT = 2_008_148
MAX_CHARS = 600
URL_TEMPLATE = "https://en.wiktionary.org/wiki/{title}"

# Each query accepts several legitimate dictionary answers.
# This is a first benchmark set, not a claim that these are the only valid answers.
CASES = [
    {
        "query": "to make something less severe, harmful, or serious",
        "expected": ["mitigate", "alleviate", "ameliorate", "attenuate"],
        "category": "everyday",
    },
    {
        "query": "a person who studies stars, planets, and other objects in space",
        "expected": ["astronomer", "astrophysicist"],
        "category": "science",
    },
    {
        "query": "an instrument used to measure temperature",
        "expected": ["thermometer", "thermoscope"],
        "category": "science",
    },
    {
        "query": "a medical doctor who diagnoses and treats illness",
        "expected": ["physician", "doctor", "clinician"],
        "category": "medicine",
    },
    {
        "query": "a two-wheeled vehicle moved by pedals",
        "expected": ["bicycle", "bike", "cycle"],
        "category": "everyday",
    },
    {
        "query": "despite that, or in spite of what was just mentioned",
        "expected": ["nevertheless", "nonetheless", "notwithstanding"],
        "category": "everyday",
    },
    {
        "query": "a very large body of salt water covering much of the Earth",
        "expected": ["ocean", "sea"],
        "category": "nature",
    },
    {
        "query": "an intense fear of enclosed or confined spaces",
        "expected": ["claustrophobia"],
        "category": "medicine",
    },
    {
        "query": "the scientific study of living organisms",
        "expected": ["biology"],
        "category": "science",
    },
    {
        "query": "the scientific study of fungi",
        "expected": ["mycology"],
        "category": "science",
    },
    {
        "query": "an animal that eats both plants and animals",
        "expected": ["omnivore"],
        "category": "nature",
    },
    {
        "query": "not allowing light to pass through; not transparent",
        "expected": ["opaque"],
        "category": "everyday",
    },
    {
        "query": "to speak quietly and unclearly, often so others cannot hear",
        "expected": ["mumble", "mutter"],
        "category": "everyday",
    },
    {
        "query": "very careful and precise about small details",
        "expected": ["meticulous", "fastidious"],
        "category": "everyday",
    },
    {
        "query": "able to recover quickly after difficulty or change",
        "expected": ["resilient", "resilience"],
        "category": "everyday",
    },
    {
        "query": "a very short period of time",
        "expected": ["moment", "instant"],
        "category": "everyday",
    },
    {
        "query": "to officially cancel or abolish a law",
        "expected": ["repeal", "revoke", "annul", "rescind"],
        "category": "law",
    },
    {
        "query": "to make a liquid weaker by adding more liquid",
        "expected": ["dilute"],
        "category": "science",
    },
    {
        "query": "certain to happen and impossible to avoid",
        "expected": ["inevitable", "unavoidable"],
        "category": "everyday",
    },
    {
        "query": "a person who writes or compiles dictionaries",
        "expected": ["lexicographer"],
        "category": "language",
    },
    {
        "query": "a structure where honeybees live and raise their young",
        "expected": ["beehive", "hive"],
        "category": "nature",
    },
    {
        "query": "the scientific study of earthquakes",
        "expected": ["seismology"],
        "category": "science",
    },
    {
        "query": "an extreme or irrational fear of heights",
        "expected": ["acrophobia"],
        "category": "medicine",
    },
    {
        "query": "to move from one region or country to another",
        "expected": ["migrate", "migration"],
        "category": "nature",
    },
    {
        "query": "the study and treatment of diseases in children",
        "expected": ["pediatrics", "paediatrics"],
        "category": "medicine",
    },
]

RUST_TEST = r'''
    #[test]
    fn benchmarks_full_dictionary_semantic_search() {
        #[derive(serde::Deserialize)]
        struct BenchmarkQuery {
            query: String,
            expected: Vec<String>,
            vector: Vec<f32>,
            category: String,
        }

        let pack_dir = std::path::PathBuf::from(
            std::env::var("ENSU_DICTIONARY_FULL_DIR").unwrap()
        );
        let query_file = std::path::PathBuf::from(
            std::env::var("ENSU_DICTIONARY_QUERY_FILE").unwrap()
        );

        let mut expected_pack = crate::config::knowledge_dataset("simplewiki")
            .expect("existing Ensu dataset configuration");

        expected_pack.current_download_identity = "dictionary-full-v1".to_owned();
        expected_pack.max_chars = 600;
        expected_pack.source_url_template =
            "https://en.wiktionary.org/wiki/{title}".to_owned();

        let index = RetrievalIndex::open(&pack_dir, &expected_pack).unwrap();

        assert_eq!(index.dataset_identity(), "dictionary-full-v1");
        assert_eq!(index.count, 2_008_148);
        assert_eq!(index.dim, 512);

        let input = std::fs::read(query_file).unwrap();
        let cases: Vec<BenchmarkQuery> =
            serde_json::from_slice(&input).unwrap();

        assert!(!cases.is_empty());

        for case in cases {
            assert_eq!(case.vector.len(), 512, "bad vector: {}", case.query);

            let hits = index.search(&case.vector, 10, 0.0).unwrap();

            let hit_rows = hits.iter().map(|hit| {
                serde_json::json!({
                    "title": hit.title,
                    "section": hit.section,
                    "score": hit.score
                })
            }).collect::<Vec<_>>();

            println!(
                "SEMANTIC_BENCH_RESULT {}",
                serde_json::json!({
                    "query": case.query,
                    "expected": case.expected,
                    "category": case.category,
                    "hits": hit_rows
                })
            );
        }
    }
'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_query_vectors(model_path: Path) -> list[dict]:
    from llama_cpp import Llama, LLAMA_POOLING_TYPE_MEAN

    actual_hash = sha256_file(model_path)
    if actual_hash != MODEL_SHA256:
        raise SystemExit(f"Embedding model SHA-256 mismatch: {actual_hash}")

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

    prompts = [
        QUERY_PROMPT.format(query=case["query"])
        for case in CASES
    ]

    embeddings = model.embed(prompts, normalize=False)

    if len(embeddings) != len(CASES):
        raise SystemExit("Embedding count does not match benchmark query count")

    output = []

    for case, embedding in zip(CASES, embeddings):
        if len(embedding) != SOURCE_DIM:
            raise SystemExit(
                f"Expected {SOURCE_DIM} dimensions for {case['query']!r}; "
                f"got {len(embedding)}"
            )

        vector = [float(value) for value in embedding[:DIM]]

        if not all(math.isfinite(value) for value in vector):
            raise SystemExit(f"Non-finite embedding: {case['query']}")

        norm = math.sqrt(math.fsum(value * value for value in vector))

        if not math.isfinite(norm) or norm <= 0:
            raise SystemExit(f"Invalid embedding norm: {case['query']}")

        vector = [value / norm for value in vector]

        output.append({
            "query": case["query"],
            "expected": case["expected"],
            "category": case["category"],
            "vector": vector,
        })

    return output


def calculate_metrics(results: list[dict]) -> dict:
    top1 = 0
    recall10 = 0
    reciprocal_rank_total = 0.0
    category_stats: dict[str, dict] = {}

    for result in results:
        expected = {
            word.casefold()
            for word in result["expected"]
        }

        titles = [
            hit["title"].strip().casefold()
            for hit in result["hits"]
        ]

        matching_ranks = [
            index + 1
            for index, title in enumerate(titles)
            if title in expected
        ]

        found = bool(matching_ranks)
        first_rank = min(matching_ranks) if found else None

        if titles and titles[0] in expected:
            top1 += 1

        if found:
            recall10 += 1
            reciprocal_rank_total += 1.0 / first_rank

        category = result["category"]
        stats = category_stats.setdefault(
            category,
            {"queries": 0, "found_at_10": 0, "top1": 0},
        )
        stats["queries"] += 1
        stats["found_at_10"] += int(found)
        stats["top1"] += int(bool(titles) and titles[0] in expected)

        result["matched"] = found
        result["first_matching_rank"] = first_rank

    total = len(results)

    return {
        "queries": total,
        "top1_accuracy": top1 / total if total else 0,
        "recall_at_10": recall10 / total if total else 0,
        "mean_reciprocal_rank": (
            reciprocal_rank_total / total if total else 0
        ),
        "top1_correct": top1,
        "found_in_top10": recall10,
        "categories": category_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("build/semantic-benchmark-report.json"),
    )
    args = parser.parse_args()

    pack = args.pack_dir.resolve()
    upstream = args.upstream_root.resolve()
    model_path = args.model.resolve()

    if not (pack / "manifest.json").is_file():
        raise SystemExit(f"Dictionary pack not found: {pack}")

    manifest = json.loads(
        (pack / "manifest.json").read_text(encoding="utf-8")
    )

    if manifest.get("dataset") != IDENTITY:
        raise SystemExit("Unexpected dictionary pack identity")
    if manifest.get("count") != COUNT or manifest.get("dim") != DIM:
        raise SystemExit("Unexpected dictionary pack dimensions")
    if manifest.get("max_chars") != MAX_CHARS:
        raise SystemExit("Unexpected dictionary max_chars")
    if manifest.get("model") != MODEL_IDENTITY:
        raise SystemExit("Dictionary pack uses an unexpected embedding model")
    if manifest.get("query_prompt") != QUERY_PROMPT:
        raise SystemExit("Dictionary pack query prompt does not match")

    if not model_path.is_file():
        raise SystemExit(f"Embedding model not found: {model_path}")

    print("Generating real query embeddings with the pinned model...", flush=True)
    cases = make_query_vectors(model_path)

    query_file = Path("build/semantic-benchmark-queries.json")
    query_file.parent.mkdir(parents=True, exist_ok=True)
    query_file.write_text(
        json.dumps(cases, ensure_ascii=False),
        encoding="utf-8",
    )

    index_rs = (
        upstream / "rust/crates/ensu/src/retrieval/index.rs"
    )
    if not index_rs.is_file():
        raise SystemExit("Ensu Rust retrieval source not found")

    marker = "#[cfg(test)]\npub(super) mod tests {"
    original = index_rs.read_text(encoding="utf-8")
    test_name = "benchmarks_full_dictionary_semantic_search"

    if marker not in original:
        raise SystemExit("Could not find Ensu RetrievalIndex test module")
    if test_name in original:
        raise SystemExit("Semantic benchmark test already exists")

    index_rs.write_text(
        original.replace(marker, marker + "\n" + RUST_TEST, 1),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["ENSU_DICTIONARY_FULL_DIR"] = str(pack)
    env["ENSU_DICTIONARY_QUERY_FILE"] = str(query_file)
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

    print("Searching all 2,008,148 dictionary passages...", flush=True)

    completed = subprocess.run(
        command,
        cwd=upstream / "rust",
        env=env,
        text=True,
        capture_output=True,
    )

    print(completed.stdout, flush=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, flush=True)

    if completed.returncode != 0:
        raise SystemExit(
            f"Rust semantic benchmark failed with exit code "
            f"{completed.returncode}"
        )

    results = []
    for line in completed.stdout.splitlines():
        marker_text = "SEMANTIC_BENCH_RESULT "
        if marker_text in line:
            payload = line.split(marker_text, 1)[1]
            results.append(json.loads(payload))

    if len(results) != len(CASES):
        raise SystemExit(
            f"Expected {len(CASES)} benchmark results; got {len(results)}"
        )

    metrics = calculate_metrics(results)

    report = {
        "dataset": IDENTITY,
        "passages": COUNT,
        "model": MODEL_IDENTITY,
        "query_prompt": QUERY_PROMPT,
        "dimensions": DIM,
        "benchmark_version": 1,
        "metrics": metrics,
        "results": results,
        "interpretation": (
            "A matching expected headword in the top 10 counts as a hit. "
            "This benchmark measures retrieval ranking, not definition "
            "completeness or dictionary editorial quality."
        ),
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nSEMANTIC_BENCHMARK_SUMMARY")
    print(json.dumps(metrics, indent=2))
    print(f"\nReport written to: {args.report}")


if __name__ == "__main__":
    main()
