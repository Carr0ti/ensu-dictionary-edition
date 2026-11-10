#!/usr/bin/env python3
"""Build and validate a tiny Ensu-compatible semantic pack pilot.

Uses the same EmbeddingGemma GGUF, prompts, mean pooling, 768 -> 512
Matryoshka prefix normalization, int8 scale, and four-file layout as Ensu.
The pilot is deliberately small; it is not the full dictionary build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any

import zstandard as zstd

MODEL_ID = "ggml-org/embeddinggemma-300M-GGUF:embeddinggemma-300M-Q8_0.gguf"
MODEL_URL = (
    "https://huggingface.co/ente-ai/embeddinggemma-300m-gguf/resolve/"
    "957b55764bd672f51240ac026e3a23ac9459ee3c/embeddinggemma-300M-Q8_0.gguf"
)
MODEL_SHA256 = "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63"
SOURCE_DIM = 768
DIM = 512
SCALE = 127
ROWS_PER_BLOCK = 128
MAX_CHARS = 600
DOC_PROMPT = "title: {title} | text: {text}"
QUERY_PROMPT = "task: search result | query: {query}"
REVISION = "dictionary-pilot-v1"
STABLE_ID = "simplewiki"
URL_TEMPLATE = "https://simple.wikipedia.org/wiki/{title}"  # temporary reader-contract fixture


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_passages(path: Path, wanted: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        title = str(item.get("title", "")).strip()
        text = str(item.get("text", "")).strip()
        if not title or not text:
            continue
        if len(text) > MAX_CHARS:
            raise ValueError(f"Passage {line_no} exceeds {MAX_CHARS} characters")
        key = title.casefold()
        if key in seen:
            continue
        seen.add(key)
        section = str(item.get("section", "")).strip()
        row = {"title": title, "text": text}
        if section:
            row["section"] = section
        rows.append(row)
    if not rows:
        raise ValueError("No usable semantic passages found")
    # Evenly sample distinct headwords across the prepared pilot, rather than
    # taking only the first few entries in source order.
    if len(rows) > wanted:
        indices = [(i * (len(rows) - 1)) // max(1, wanted - 1) for i in range(wanted)]
        rows = [rows[i] for i in indices]
    return rows


def load_embedder(model_path: Path, threads: int):
    try:
        from llama_cpp import Llama, LLAMA_POOLING_TYPE_MEAN
    except Exception as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. Install the pinned CPU wheel first."
        ) from exc
    return Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_batch=2048,
        n_ubatch=2048,
        n_threads=threads,
        n_threads_batch=threads,
        embedding=True,
        pooling_type=LLAMA_POOLING_TYPE_MEAN,
        n_gpu_layers=0,
        verbose=False,
    )


def get_embedding(llm, prompt: str) -> list[float]:
    # llama-cpp-python's embed() returns either one vector or a batch of vectors
    # depending on the pinned wheel/API. Normalization is repeated below on the
    # exact first 512 components, so the raw/full-vector norm is not relied on.
    if hasattr(llm, "embed"):
        try:
            result = llm.embed(prompt, normalize=False)
        except TypeError:
            result = llm.embed(prompt)
        if result and isinstance(result[0], (list, tuple)):
            vector = result[0]
        else:
            vector = result
    else:
        result = llm.create_embedding(prompt)
        vector = result["data"][0]["embedding"]
    vector = [float(x) for x in vector]
    if len(vector) != SOURCE_DIM:
        raise ValueError(
            f"Embedding model returned {len(vector)} dimensions; expected {SOURCE_DIM}"
        )
    if not all(math.isfinite(x) for x in vector):
        raise ValueError("Embedding contains NaN or infinity")
    return vector


def normalize_prefix(vector: list[float]) -> list[float]:
    prefix = vector[:DIM]
    norm = math.sqrt(math.fsum(x * x for x in prefix))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding has zero or invalid norm")
    return [x / norm for x in prefix]


def quantize(vector: list[float]) -> bytes:
    # Store signed int8 values as their two's-complement bytes.
    values = []
    for x in normalize_prefix(vector):
        q = max(-127, min(127, int(round(x * SCALE))))
        values.append(q & 0xFF)
    return bytes(values)


def signed_bytes(data: bytes) -> list[int]:
    return [b if b < 128 else b - 256 for b in data]


def write_pack(rows: list[dict[str, str]], vectors: list[bytes], out_dir: Path) -> None:
    if len(rows) != len(vectors):
        raise ValueError("Metadata/vector row count mismatch")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vectors.i8").write_bytes(b"".join(vectors))

    compressor = zstd.ZstdCompressor(level=19)
    offsets = [0]
    with (out_dir / "meta.zst").open("wb") as meta:
        for start in range(0, len(rows), ROWS_PER_BLOCK):
            block = rows[start:start + ROWS_PER_BLOCK]
            payload = "".join(
                json.dumps({**{"title": row["title"], "text": row["text"]}, **({"section": row["section"]} if row.get("section") else {})},
                           ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in block
            ).encode("utf-8")
            frame = compressor.compress(payload)
            meta.write(frame)
            offsets.append(meta.tell())

    (out_dir / "meta.offsets").write_bytes(
        b"".join(struct.pack("<Q", offset) for offset in offsets)
    )

    manifest = {
        "dataset": REVISION,
        "granularity": "one-passage-per-sampled-headword",
        "max_chars": MAX_CHARS,
        "model": MODEL_ID,
        "source_dim": SOURCE_DIM,
        "matryoshka": True,
        "doc_prompt": DOC_PROMPT,
        "query_prompt": QUERY_PROMPT,
        "dim": DIM,
        "count": len(rows),
        "quant": "int8",
        "scale": SCALE,
        "meta_codec": "zstd-blocks",
        "meta_rows_per_block": ROWS_PER_BLOCK,
        "url_template": URL_TEMPLATE,
        "attribution": "Wiktionary contributors; extracted with Wiktextract",
        "license": "CC BY-SA 4.0; retain source attribution and share-alike terms",
        "model_sha256": MODEL_SHA256,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_pack(pack_dir: Path) -> dict[str, Any]:
    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    required = {
        "model": MODEL_ID, "source_dim": SOURCE_DIM, "matryoshka": True,
        "doc_prompt": DOC_PROMPT, "query_prompt": QUERY_PROMPT, "dim": DIM,
        "quant": "int8", "scale": SCALE, "meta_codec": "zstd-blocks",
        "meta_rows_per_block": ROWS_PER_BLOCK, "max_chars": MAX_CHARS,
        "dataset": REVISION, "url_template": URL_TEMPLATE,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Manifest mismatch for {key}: {manifest.get(key)!r}")
    count = int(manifest["count"])
    if count <= 0:
        raise ValueError("Manifest count must be positive")

    vectors_path = pack_dir / "vectors.i8"
    meta_path = pack_dir / "meta.zst"
    offsets_path = pack_dir / "meta.offsets"
    if vectors_path.stat().st_size != count * DIM:
        raise ValueError("vectors.i8 length does not equal count × dim")

    block_count = (count + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
    raw_offsets = offsets_path.read_bytes()
    if len(raw_offsets) != (block_count + 1) * 8:
        raise ValueError("meta.offsets has the wrong byte length")
    offsets = list(struct.unpack("<" + "Q" * (block_count + 1), raw_offsets))
    meta_size = meta_path.stat().st_size
    if offsets[0] != 0 or offsets[-1] != meta_size:
        raise ValueError("meta.offsets endpoints are invalid")
    if any(a >= b for a, b in zip(offsets, offsets[1:])):
        raise ValueError("meta.offsets must be strictly increasing")

    dctx = zstd.ZstdDecompressor()
    metadata_rows = []
    with meta_path.open("rb") as f:
        for block in range(block_count):
            start, end = offsets[block], offsets[block + 1]
            f.seek(start)
            frame = f.read(end - start)
            decoded = dctx.decompress(frame).decode("utf-8")
            lines = decoded.splitlines()
            expected_rows = min(ROWS_PER_BLOCK, count - block * ROWS_PER_BLOCK)
            if len(lines) != expected_rows:
                raise ValueError(f"Metadata block {block} row count mismatch")
            for line in lines:
                row = json.loads(line)
                if set(row) - {"title", "text", "section"}:
                    raise ValueError(f"Metadata row has unsupported fields: {sorted(set(row) - {'title', 'text', 'section'})}")
                if not isinstance(row.get("title"), str) or not row["title"].strip():
                    raise ValueError("Metadata row has an empty or non-string title")
                if not isinstance(row.get("text"), str) or not row["text"].strip():
                    raise ValueError("Metadata row has an empty or non-string text")
                if "section" in row and (not isinstance(row["section"], str) or not row["section"].strip()):
                    raise ValueError("Metadata section must be a non-empty string when present")
                if len(row["text"]) > MAX_CHARS:
                    raise ValueError("Metadata passage exceeds max_chars")
                metadata_rows.append(row)

    vector_data = vectors_path.read_bytes()
    # The quantized vectors should remain close to unit length after scaling.
    norms = []
    for i in range(count):
        signed = signed_bytes(vector_data[i * DIM:(i + 1) * DIM])
        norms.append(math.sqrt(math.fsum((x / SCALE) ** 2 for x in signed)))

    return {
        "validated": True,
        "revision": manifest["dataset"],
        "count": count,
        "dim": DIM,
        "vector_bytes": vectors_path.stat().st_size,
        "metadata_bytes": meta_size,
        "offset_bytes": offsets_path.stat().st_size,
        "metadata_rows": len(metadata_rows),
        "min_quantized_vector_norm": min(norms),
        "max_quantized_vector_norm": max(norms),
        "max_passage_chars": max(len(row["text"]) for row in metadata_rows),
        "titles": [row["title"] for row in metadata_rows],
        "metadata_fields_validated": True,
        "metadata_schema": "{title: string, text: string, section?: string}",
    }


def run_retrieval_smoke(llm, rows: list[dict[str, str]], vectors: list[bytes]) -> list[dict]:
    # Query using Ensu's exact query prompt. This is a smoke test, not a
    # benchmark or a claim that every headword should rank first.
    tests = []
    for row in rows[:min(5, len(rows))]:
        prompt = QUERY_PROMPT.format(query=f"What does {row['title']} mean?")
        query = normalize_prefix(get_embedding(llm, prompt))
        scored = []
        for candidate, raw in zip(rows, vectors):
            signed = signed_bytes(raw)
            score = math.fsum(q * (v / SCALE) for q, v in zip(query, signed))
            scored.append((score, candidate["title"]))
        scored.sort(reverse=True)
        tests.append({
            "query": f"What does {row['title']} mean?",
            "expected_headword": row["title"],
            "top_results": [{"title": title, "score": round(score, 5)}
                            for score, title in scored[:5]],
            "expected_in_top_5": any(title.casefold() == row["title"].casefold()
                                     for _, title in scored[:5]),
        })
    return tests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", default="build/semantic-pilot/pilot.jsonl")
    ap.add_argument("--model", default="build/models/embeddinggemma-300M-Q8_0.gguf")
    ap.add_argument("--output", default="build/semantic-pack/dictionary/dictionary-pilot-v1")
    ap.add_argument("--passages", type=int, default=32)
    ap.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 2)))
    args = ap.parse_args()

    input_path = Path(args.input_jsonl)
    model_path = Path(args.model)
    if not input_path.is_file():
        raise SystemExit(f"Pilot JSONL not found: {input_path}")
    if not model_path.is_file():
        raise SystemExit(f"Embedding model not found: {model_path}")
    actual_sha = sha256_file(model_path)
    if actual_sha != MODEL_SHA256:
        raise SystemExit(f"Embedding model SHA-256 mismatch: {actual_sha}")

    rows = read_passages(input_path, args.passages)
    print(f"Embedding {len(rows)} passages with Ensu's pinned EmbeddingGemma model...")
    llm = load_embedder(model_path, args.threads)
    vectors = []
    for index, row in enumerate(rows, 1):
        prompt = DOC_PROMPT.format(title=row["title"], text=row["text"])
        vectors.append(quantize(get_embedding(llm, prompt)))
        print(f"Embedded {index}/{len(rows)}: {row['title']}", flush=True)

    out_dir = Path(args.output)
    write_pack(rows, vectors, out_dir)
    report = validate_pack(out_dir)
    report["model_sha256"] = actual_sha
    report["retrieval_smoke_tests"] = run_retrieval_smoke(llm, rows, vectors)
    report["files"] = {
        name: {"bytes": (out_dir / name).stat().st_size,
               "sha256": sha256_file(out_dir / name)}
        for name in ("manifest.json", "vectors.i8", "meta.zst", "meta.offsets")
    }
    report_path = out_dir / "pilot-pack-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "titles"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
