
#!/usr/bin/env python3
"""Benchmark definition-to-headword retrieval using words actually present in the 10K pack."""

import argparse
import hashlib
import io
import json
import pathlib
import re
import shutil
import subprocess
import urllib.request
import zipfile

MODEL_URL = (
    "https://huggingface.co/ente-ai/embeddinggemma-300m-gguf/resolve/"
    "957b55764bd672f51240ac026e3a23ac9459ee3c/"
    "embeddinggemma-300M-Q8_0.gguf"
)

MODEL_SHA256 = (
    "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63"
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-zip", required=True)
    ap.add_argument(
        "--model",
        default="embeddinggemma-300M-Q8_0.gguf",
    )
    ap.add_argument("--upstream", default="upstream")
    ap.add_argument("--cases", type=int, default=40)
    args = ap.parse_args()

    zpath = pathlib.Path(args.pack_zip)

    if not zpath.exists():
        raise SystemExit(f"Pack ZIP missing: {zpath}")

    upstream = pathlib.Path(args.upstream)
    rust_root = upstream / "rust"

    pack = (
        rust_root
        / "target/ensu-dictionary-10k/dictionary-10k-v1"
    )

    if pack.exists():
        shutil.rmtree(pack)

    pack.mkdir(parents=True)

    # Extract the four files required by Ensu.
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()

        for filename in (
            "manifest.json",
            "vectors.i8",
            "meta.zst",
            "meta.offsets",
        ):
            matches = [
                n for n in names
                if n == filename or n.endswith("/" + filename)
            ]

            if not matches:
                raise SystemExit(
                    f"Missing {filename} in {zpath}"
                )

            (pack / filename).write_bytes(
                z.read(matches[0])
            )

    manifest = json.loads(
        (pack / "manifest.json").read_text()
    )

    if manifest.get("dataset") != "dictionary-10k-v1":
        raise SystemExit(
            f"Unexpected dataset identity: {manifest.get('dataset')}"
        )

    # Read every concatenated Zstandard metadata frame.
    import zstandard as zstd

    compressed = (pack / "meta.zst").read_bytes()

    with zstd.ZstdDecompressor().stream_reader(
        io.BytesIO(compressed),
        read_across_frames=True,
    ) as reader:
        raw = reader.read()

    rows = [
        json.loads(line)
        for line in raw.splitlines()
        if line.strip()
    ]

    # Build queries from real definitions, removing the headword.
    candidates = []
    seen = set()

    for row in rows:
        title = str(row.get("title", "")).strip()
        body = str(row.get("text", "")).strip()

        key = title.casefold()

        if not title or not body or key in seen:
            continue

        seen.add(key)

        # FIX 1: Remove the complete headword/phrase only.
        pattern = re.compile(
            r"(?<!\w)" + re.escape(title) + r"(?!\w)",
            flags=re.IGNORECASE | re.UNICODE,
        )

        clean = pattern.sub(" ", body)
        clean = re.sub(r"\s+", " ", clean).strip(" |;,:.-")

        # Verify that the complete headword was removed.
        if pattern.search(clean):
            raise SystemExit(
                f"Headword-removal check failed for {title!r}"
            )

        if len(clean) < 45:
            continue

        candidates.append({
            "expected": title,
            "query": "Find the word that means: " + clean[:420],
        })

    if len(candidates) < 10:
        raise SystemExit(
            f"Not enough usable passages for benchmark: {len(candidates)}"
        )

    count = min(args.cases, len(candidates))

    # Spread test cases across the whole pack.
    if count > 1:
        selected = [
            candidates[
                round(i * (len(candidates) - 1) / (count - 1))
            ]
            for i in range(count)
        ]
    else:
        selected = [candidates[0]]

    print(
        f"Pack passages: {len(rows)}; "
        f"usable definition cases: {len(candidates)}; "
        f"selected: {len(selected)}",
        flush=True,
    )

    # Download and verify the exact pinned embedding model.
    model = pathlib.Path(args.model)

    if not model.exists():
        print(
            "Downloading pinned EmbeddingGemma model...",
            flush=True,
        )
        urllib.request.urlretrieve(MODEL_URL, model)

    if sha256(model) != MODEL_SHA256:
        raise SystemExit(
            "Embedding model SHA256 mismatch"
        )

    from llama_cpp import Llama

    llm = Llama(
        model_path=str(model),
        embedding=True,
        n_ctx=2048,
        n_threads=2,
        verbose=False,
    )

    # Generate Ensu-compatible query embeddings.
    cases = []

    for case in selected:
        prompt = (
            f"task: search result | query: {case['query']}"
        )

        embedding = (
            llm.create_embedding(prompt)["data"][0]["embedding"][:512]
        )

        norm = sum(x * x for x in embedding) ** 0.5

        vector = [
            float(x / norm)
            for x in embedding
        ]

        cases.append({
            **case,
            "vector": vector,
        })

    cases_json = json.dumps(
        cases,
        separators=(",", ":"),
    )

    # This test runs from the Rust package directory.
    # CARGO_MANIFEST_DIR makes the pack path independent
    # of Cargo's test working directory.
    rust_test = r'''
#[test]
fn benchmarks_dictionary_10k_semantic_queries() {
    let cfg = crate::config::knowledge_dataset("simplewiki")
        .expect("simplewiki config");

    let pack_path = std::path::Path::new(
        env!("CARGO_MANIFEST_DIR")
    )
    .join("../../target/ensu-dictionary-10k/dictionary-10k-v1");

    let index = super::RetrievalIndex::open(
        &pack_path,
        &cfg
    )
    .expect("open 10K pack");

    let cases: serde_json::Value =
        serde_json::from_str(r#"CASES_JSON"#).unwrap();

    let mut top1 = 0usize;
    let mut top5 = 0usize;
    let mut total = 0usize;

    for c in cases.as_array().unwrap() {
        let query = c["query"].as_str().unwrap();

        let expected = c["expected"]
            .as_str()
            .unwrap()
            .to_lowercase();

        let vector: Vec<f32> = c["vector"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();

        let hits = index
            .search(&vector, 5, 0.0)
            .expect("search query");

        // FIX 2: Compare actual result titles, not Debug output.
        let one = hits
            .first()
            .map(|hit| hit.title.to_lowercase() == expected)
            .unwrap_or(false);

        let five = hits
            .iter()
            .any(|hit| hit.title.to_lowercase() == expected);

        if one {
            top1 += 1;
        }

        if five {
            top5 += 1;
        }

        total += 1;

        println!(
            "BENCH|{}|{}|{}|{}",
            query,
            expected,
            if one { "TOP1" } else { "MISS_TOP1" },
            if five { "TOP5" } else { "MISS_TOP5" }
        );
    }

    println!(
        "BENCH_SUMMARY|{}|{}|{}",
        total,
        top1,
        top5
    );

    assert!(
        total >= 10,
        "not enough benchmark cases"
    );
}
'''.replace("CASES_JSON", cases_json)

    # Temporarily add the benchmark test to Ensu's real Rust reader.
    source = (
        rust_root
        / "crates/ensu/src/retrieval/index.rs"
    )

    if not source.exists():
        raise SystemExit(
            f"Ensu source not found: {source}"
        )

    original = source.read_text()

    marker = (
        "#[cfg(test)]\npub(super) mod tests {"
    )

    if marker not in original:
        raise SystemExit(
            "Could not find Ensu retrieval test module marker"
        )

    source.write_text(
        original.replace(
            marker,
            marker + "\n" + rust_test,
            1,
        )
    )

    try:
        command = [
            "cargo",
            "test",
            "-p",
            "ente-ensu",
            "--lib",
            "--no-default-features",
            "retrieval::index::tests::benchmarks_dictionary_10k_semantic_queries",
            "--",
            "--nocapture",
        ]

        print(
            "Running definition-to-headword benchmark "
            "through Ensu Rust RetrievalIndex...",
            flush=True,
        )

        result = subprocess.run(
            command,
            cwd=rust_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        print(result.stdout)

        if result.returncode:
            raise SystemExit(result.returncode)

        match = re.search(
            r"BENCH_SUMMARY\|(\d+)\|(\d+)\|(\d+)",
            result.stdout,
        )

        if not match:
            raise SystemExit(
                "Benchmark summary not found in Rust output"
            )

        total, top1, top5 = map(
            int,
            match.groups(),
        )

        report = {
            "dataset": "dictionary-10k-v1",
            "benchmark_type": (
                "definition-to-headword retrieval; "
                "queries derived from passage definitions; "
                "whole-headword removal verified; exact hit.title matching"
            ),
            "cases": total,
            "top1_correct": top1,
            "top5_correct": top5,
            "top1_accuracy": round(top1 / total, 4),
            "top5_accuracy": round(top5 / total, 4),
        }

        pathlib.Path(
            "semantic-quality-report.json"
        ).write_text(
            json.dumps(report, indent=2) + "\n"
        )

        print(
            "QUALITY_REPORT=" + json.dumps(report),
            flush=True,
        )

    finally:
        # Always restore Ensu's original source file.
        source.write_text(original)


if __name__ == "__main__":
    main()
