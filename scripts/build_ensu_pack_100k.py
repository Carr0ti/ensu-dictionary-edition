#!/usr/bin/env python3
"""Build a 100,000-passage Ensu-compatible dictionary retrieval pack."""
from __future__ import annotations
import argparse, hashlib, json, math, os, struct
from pathlib import Path
import zstandard as zstd

MODEL_ID = "ggml-org/embeddinggemma-300M-GGUF:embeddinggemma-300M-Q8_0.gguf"
MODEL_SHA256 = "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63"
SOURCE_DIM, DIM, SCALE, ROWS_PER_BLOCK = 768, 512, 127, 128
MAX_CHARS = 600
DOC_PROMPT = "title: {title} | text: {text}"
REVISION = "dictionary-100k-v1"
URL_TEMPLATE = "https://simple.wikipedia.org/wiki/{title}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path, wanted: int) -> list[dict[str, str]]:
    rows, seen = [], set()
    with path.open(encoding="utf-8") as src:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            title, text = str(item.get("title", "")).strip(), str(item.get("text", "")).strip()
            if not title or not text:
                continue
            if len(text) > MAX_CHARS:
                raise ValueError(f"Passage {line_no} exceeds {MAX_CHARS} characters")
            key = title.casefold()
            if key in seen:
                continue
            seen.add(key)
            row = {"title": title, "text": text}
            section = str(item.get("section", "")).strip()
            if section:
                row["section"] = section
            rows.append(row)
    if len(rows) < wanted:
        raise SystemExit(f"Only {len(rows)} distinct headwords available; need {wanted}. Increase preparation --entries.")
    if wanted == 1:
        rows = [rows[len(rows) // 2]]
    elif len(rows) > wanted:
        indices = [(i * (len(rows) - 1)) // (wanted - 1) for i in range(wanted)]
        rows = [rows[i] for i in indices]
    return rows


def load_model(path: Path, threads: int):
    from llama_cpp import Llama, LLAMA_POOLING_TYPE_MEAN
    return Llama(model_path=str(path), n_ctx=2048, n_batch=2048, n_ubatch=2048,
                 n_threads=threads, n_threads_batch=threads, embedding=True,
                 pooling_type=LLAMA_POOLING_TYPE_MEAN, n_gpu_layers=0, verbose=False)


def embed(llm, text: str) -> list[float]:
    try:
        result = llm.embed(text, normalize=False)
    except TypeError:
        result = llm.embed(text)
    vector = result[0] if result and isinstance(result[0], (list, tuple)) else result
    vector = [float(x) for x in vector]
    if len(vector) != SOURCE_DIM or not all(math.isfinite(x) for x in vector):
        raise ValueError("Model returned invalid embedding dimensions or values")
    prefix = vector[:DIM]
    norm = math.sqrt(math.fsum(x*x for x in prefix))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding has zero or invalid norm")
    return [x / norm for x in prefix]


def quantize(vector: list[float]) -> bytes:
    return bytes((max(-127, min(127, int(round(x * SCALE)))) & 0xff) for x in vector)


def write_pack(rows, vectors, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / "vectors.i8").write_bytes(b"".join(vectors))
    offsets = [0]
    compressor = zstd.ZstdCompressor(level=9)
    with (out / "meta.zst").open("wb") as f:
        for start in range(0, len(rows), ROWS_PER_BLOCK):
            block = rows[start:start + ROWS_PER_BLOCK]
            payload = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in block).encode()
            f.write(compressor.compress(payload))
            offsets.append(f.tell())
    (out / "meta.offsets").write_bytes(b"".join(struct.pack("<Q", x) for x in offsets))
    manifest = {
        "dataset": REVISION, "granularity": "one-passage-per-headword",
        "max_chars": MAX_CHARS, "model": MODEL_ID, "source_dim": SOURCE_DIM,
        "matryoshka": True, "doc_prompt": DOC_PROMPT,
        "query_prompt": "task: search result | query: {query}",
        "dim": DIM, "count": len(rows), "quant": "int8", "scale": SCALE,
        "meta_codec": "zstd-blocks", "meta_rows_per_block": ROWS_PER_BLOCK,
        "url_template": URL_TEMPLATE,
        "attribution": "Wiktionary contributors; extracted with Wiktextract",
        "license": "CC BY-SA 4.0; retain source attribution and share-alike terms",
        "model_sha256": MODEL_SHA256,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate(out: Path, expected: int):
    m = json.loads((out / "manifest.json").read_text())
    assert m["dataset"] == REVISION and m["count"] == expected and m["dim"] == DIM
    assert (out / "vectors.i8").stat().st_size == expected * DIM
    block_count = (expected + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
    raw = (out / "meta.offsets").read_bytes()
    assert len(raw) == (block_count + 1) * 8
    offsets = struct.unpack("<" + "Q" * (block_count + 1), raw)
    meta_path = out / "meta.zst"
    assert offsets[0] == 0 and offsets[-1] == meta_path.stat().st_size
    assert all(a < b for a, b in zip(offsets, offsets[1:]))
    rows_seen = 0
    dctx = zstd.ZstdDecompressor()
    with meta_path.open("rb") as f:
        for i in range(block_count):
            f.seek(offsets[i]); data = f.read(offsets[i+1] - offsets[i])
            lines = dctx.decompress(data).decode("utf-8").splitlines()
            assert len(lines) == min(ROWS_PER_BLOCK, expected - i * ROWS_PER_BLOCK)
            for line in lines:
                row = json.loads(line)
                assert row.get("title") and row.get("text") and len(row["text"]) <= MAX_CHARS
                rows_seen += 1
    assert rows_seen == expected
    packed_rows = read_rows_from_pack(out)
    report = {
        "validated": True, "revision": REVISION, "passage_count": expected,
        "unique_headwords": len({r["title"].casefold() for r in packed_rows}),
        "dimension": DIM, "vector_bytes": (out / "vectors.i8").stat().st_size,
        "metadata_bytes": meta_path.stat().st_size, "offset_bytes": (out / "meta.offsets").stat().st_size,
        "metadata_blocks": block_count, "metadata_rows_validated": rows_seen,
        "max_passage_chars": max(len(r["text"]) for r in packed_rows),
        "files": {n: {"bytes": (out/n).stat().st_size, "sha256": sha256_file(out/n)} for n in ("manifest.json", "vectors.i8", "meta.zst", "meta.offsets")},
    }
    (out / "pack-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def read_rows_from_pack(out: Path):
    m = json.loads((out / "manifest.json").read_text())
    count = m["count"]; blocks = (count + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
    offsets = struct.unpack("<" + "Q" * (blocks + 1), (out / "meta.offsets").read_bytes())
    result=[]; dctx=zstd.ZstdDecompressor()
    with (out / "meta.zst").open("rb") as f:
        for i in range(blocks):
            f.seek(offsets[i]); lines=dctx.decompress(f.read(offsets[i+1]-offsets[i])).decode().splitlines()
            result.extend(json.loads(x) for x in lines)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", default="build/semantic-100k/pilot.jsonl")
    ap.add_argument("--model", default="build/models/embeddinggemma-300M-Q8_0.gguf")
    ap.add_argument("--output", default="build/semantic-pack/dictionary/dictionary-100k-v1")
    ap.add_argument("--passages", type=int, default=100000)
    ap.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 2)))
    a=ap.parse_args()
    if a.passages < 1:
        raise SystemExit("--passages must be at least 1")
    model=Path(a.model)
    if not model.is_file(): raise SystemExit(f"Model not found: {model}")
    digest=sha256_file(model)
    if digest != MODEL_SHA256: raise SystemExit(f"Model SHA-256 mismatch: {digest}")
    rows=read_rows(Path(a.input_jsonl), a.passages)
    print(f"Embedding {len(rows):,} unique headwords...", flush=True)
    llm=load_model(model, a.threads); vectors=[]
    for i,row in enumerate(rows,1):
        vectors.append(quantize(embed(llm, DOC_PROMPT.format(title=row["title"], text=row["text"]))))
        if i % 100 == 0 or i == len(rows): print(f"Embedded {i:,}/{len(rows):,}", flush=True)
    out=Path(a.output); write_pack(rows,vectors,out)
    report=validate(out,a.passages); report["model_sha256"]=digest
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__ == "__main__": main()
