#!/usr/bin/env python3
"""Create real EmbeddingGemma query vectors for the existing Ensu pilot pack."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import zstandard as zstd

MODEL_SHA256 = "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63"
SOURCE_DIM = 768
DIM = 512
QUERY_PROMPT = "task: search result | query: {query}"

# Queries deliberately describe a meaning rather than repeating the headword.
TEST_QUERIES = [
    ("dictionary", "a reference work that lists words alphabetically and explains their meanings, pronunciation, and usage"),
    ("kyu", "a grade or rank used in Japanese martial arts and traditional activities"),
    ("dealing", "a commercial transaction or exchange between people or businesses"),
    ("let out", "to release or allow someone or something to leave or escape"),
    ("design code", "a set of rules governing the design of buildings and public spaces"),
    ("beach bag", "a carryall used to bring towels and other belongings on a seaside outing"),
    ("pneumologist", "a medical specialist who diagnoses and treats diseases of the lungs"),
    ("blue won", "a foreign exchange certificate issued to visitors travelling to North Korea"),
    ("ignoramo", "a foolish person who lacks knowledge or good judgment"),
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def normalize_prefix(vector):
    prefix = [float(x) for x in vector[:DIM]]
    norm = math.sqrt(math.fsum(x*x for x in prefix))
    if len(vector) != SOURCE_DIM or not math.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding has invalid dimensions or norm")
    return [x / norm for x in prefix]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-zip", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    if sha256_file(args.model) != MODEL_SHA256:
        raise SystemExit("Embedding model SHA-256 mismatch")

    import zipfile
    with zipfile.ZipFile(args.pilot_zip) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        if manifest.get("dataset") != "dictionary-pilot-v1" or manifest.get("dim") != DIM:
            raise SystemExit("Unexpected pilot pack identity or vector dimension")
        decoded = zstd.ZstdDecompressor().decompress(zf.read("meta.zst")).decode("utf-8")
    rows = [json.loads(line) for line in decoded.splitlines() if line.strip()]
    by_title = {row["title"].casefold(): row for row in rows}
    missing = [title for title, _ in TEST_QUERIES if title.casefold() not in by_title]
    if missing:
        raise SystemExit(f"Pilot pack is missing expected test headwords: {missing}")

    from llama_cpp import Llama, LLAMA_POOLING_TYPE_MEAN
    llm = Llama(
        model_path=str(args.model), n_ctx=2048, n_batch=2048, n_ubatch=2048,
        n_threads=args.threads, n_threads_batch=args.threads,
        embedding=True, pooling_type=LLAMA_POOLING_TYPE_MEAN,
        n_gpu_layers=0, verbose=False,
    )
    fixtures = []
    for number, (expected, query_text) in enumerate(TEST_QUERIES, 1):
        prompt = QUERY_PROMPT.format(query=query_text)
        try:
            result = llm.embed(prompt, normalize=False)
        except TypeError:
            result = llm.embed(prompt)
        vector = result[0] if result and isinstance(result[0], (list, tuple)) else result
        normalized = normalize_prefix(vector)
        fixtures.append({
            "query": query_text,
            "expected_headword": expected,
            "vector": normalized,
        })
        print(f"Embedded query {number}/{len(TEST_QUERIES)}: {query_text}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixtures, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(fixtures)} normalized {DIM}-dimension query vectors to {args.output}")

if __name__ == "__main__":
    main()
