#!/usr/bin/env python3
"""Embed a small semantic-query set and exercise Ensu's real Rust RetrievalIndex."""
import argparse, json, os, pathlib, re, shutil, subprocess, sys, urllib.request

MODEL_URL = "https://huggingface.co/ente-ai/embeddinggemma-300m-gguf/resolve/957b55764bd672f51240ac026e3a23ac9459ee3c/embeddinggemma-300M-Q8_0.gguf"
MODEL_SHA256 = "b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63"
CASES = [
    {"query":"a person who treats sick people", "expected":"doctor"},
    {"query":"a medical doctor who specializes in the lungs", "expected":"pneumologist"},
    {"query":"the opposite of happy; feeling sad", "expected":"sad"},
    {"query":"a large body of salt water", "expected":"ocean"},
    {"query":"a place with sand beside the sea", "expected":"beach"},
    {"query":"a container used to carry personal belongings", "expected":"bag"},
    {"query":"to let someone leave or set them free", "expected":"release"},
    {"query":"to create a plan or draw up how something should look", "expected":"design"},
    {"query":"a person who writes computer programs", "expected":"programmer"},
    {"query":"the activity of writing instructions for a computer", "expected":"programming"},
    {"query":"a building where people live", "expected":"house"},
    {"query":"a vehicle with two wheels powered by pedals", "expected":"bicycle"},
    {"query":"a very large area of trees and woodland", "expected":"forest"},
    {"query":"a person who studies the stars and planets", "expected":"astronomer"},
    {"query":"a sudden strong feeling of fear", "expected":"terror"},
    {"query":"to look at something carefully to understand it", "expected":"examine"},
    {"query":"a word that means the same as another word", "expected":"synonym"},
    {"query":"a word with the opposite meaning of another word", "expected":"antonym"},
    {"query":"the process of learning or gaining knowledge", "expected":"education"},
    {"query":"a person who designs buildings", "expected":"architect"},
    {"query":"a small domesticated animal that meows", "expected":"cat"},
    {"query":"a domesticated animal that barks", "expected":"dog"},
    {"query":"a place where books are kept for reading or borrowing", "expected":"library"},
    {"query":"a device used to take photographs", "expected":"camera"},
    {"query":"a period of seven days", "expected":"week"},
    {"query":"a person who teaches students", "expected":"teacher"},
    {"query":"to move quickly on foot", "expected":"run"},
    {"query":"a state of being very thirsty", "expected":"thirst"},
    {"query":"the opposite of increase", "expected":"decrease"},
    {"query":"a written account of words and their meanings", "expected":"dictionary"},
]

def sha256(path):
    import hashlib
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pack-zip', required=True)
    ap.add_argument('--model', default='embeddinggemma-300M-Q8_0.gguf')
    ap.add_argument('--upstream', default='upstream')
    ap.add_argument('--repo', default='.')
    args=ap.parse_args()
    import zipfile, tempfile
    zpath=pathlib.Path(args.pack_zip)
    if not zpath.exists(): raise SystemExit(f'Pack ZIP missing: {zpath}')
    work=pathlib.Path(args.upstream)
    pack=work/'target/ensu-dictionary-10k/dictionary-10k-v1'
    if pack.exists(): shutil.rmtree(pack)
    pack.mkdir(parents=True)
    with zipfile.ZipFile(zpath) as z:
        names=z.namelist()
        for fn in ('manifest.json','vectors.i8','meta.zst','meta.offsets'):
            matches=[n for n in names if n.endswith('/'+fn) or n==fn]
            if not matches: raise SystemExit(f'Missing {fn} in {zpath}')
            (pack/fn).write_bytes(z.read(matches[0]))
    manifest=json.loads((pack/'manifest.json').read_text())
    if manifest.get('dataset')!='dictionary-10k-v1': raise SystemExit('Unexpected dataset identity')
    if not pathlib.Path(args.model).exists():
        print('Downloading pinned EmbeddingGemma model...', flush=True)
        urllib.request.urlretrieve(MODEL_URL,args.model)
    if sha256(args.model)!=MODEL_SHA256: raise SystemExit('Embedding model SHA256 mismatch')
    import zstandard as zstd
    rows=[]
    raw=zstd.ZstdDecompressor().decompress((pack/'meta.zst').read_bytes())
    for line in raw.splitlines():
        if line.strip(): rows.append(json.loads(line))
    titles={str(r.get('title','')).casefold() for r in rows}
    active=[c for c in CASES if c['expected'].casefold() in titles]
    skipped=[c for c in CASES if c not in active]
    if len(active)<5: raise SystemExit(f'Only {len(active)} expected headwords exist in this sampled pack; need at least 5. Missing: {[x["expected"] for x in skipped]}')
    print(f'Available benchmark cases: {len(active)}; skipped absent headwords: {len(skipped)}',flush=True)
    from llama_cpp import Llama
    llm=Llama(model_path=args.model, embedding=True, n_ctx=2048, n_threads=2, verbose=False)
    vectors=[]
    for c in active:
        prompt=f"task: search result | query: {c['query']}"
        emb=llm.create_embedding(prompt)['data'][0]['embedding'][:512]
        norm=sum(x*x for x in emb)**0.5
        vectors.append([float(x/norm) for x in emb])
    # Rust test prints ranked debug records; the Python runner parses pass/fail from output.
    cases_json=json.dumps([{**c,'vector':v} for c,v in zip(active,vectors)],separators=(',',':'))
    rust_test=r'''
#[test]
fn benchmarks_dictionary_10k_semantic_queries() {
    let cfg = crate::config::knowledge_dataset("simplewiki").expect("simplewiki config");
    let index = super::RetrievalIndex::open("target/ensu-dictionary-10k/dictionary-10k-v1", &cfg).expect("open 10K pack");
    let cases: serde_json::Value = serde_json::from_str(r#"CASES_JSON"#).unwrap();
    let mut top1 = 0usize; let mut top5 = 0usize; let mut total = 0usize;
    for c in cases.as_array().unwrap() {
        let query = c["query"].as_str().unwrap();
        let expected = c["expected"].as_str().unwrap().to_lowercase();
        let vector: Vec<f32> = c["vector"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let hits = index.search(&vector, 5, 0.0).expect("search query");
        let debug: Vec<String> = hits.iter().map(|h| format!("{:?}", h).to_lowercase()).collect();
        let one = debug.first().map(|s| s.contains(&expected)).unwrap_or(false);
        let five = debug.iter().any(|s| s.contains(&expected));
        if one { top1 += 1; } if five { top5 += 1; } total += 1;
        println!("BENCH|{}|{}|{}|{}", query, expected, if one {"TOP1"} else {"MISS_TOP1"}, if five {"TOP5"} else {"MISS_TOP5"});
    }
    println!("BENCH_SUMMARY|{}|{}|{}", total, top1, top5);
    assert!(total >= 5, "not enough benchmark cases");
}
'''.replace('CASES_JSON',cases_json)
    src=pathlib.Path(args.upstream)/'rust/crates/ensu/src/retrieval/index.rs'
    if not src.exists(): raise SystemExit(f'Ensu source not found: {src}')
    original=src.read_text()
    marker='#[cfg(test)]\npub(super) mod tests {'
    if marker not in original: raise SystemExit('Could not find Ensu retrieval test module marker')
    # serde_json is already used by Ensu tests; insert as a sibling test in existing module.
    injected=original.replace(marker, marker+'\n'+rust_test,1)
    src.write_text(injected)
    try:
        cmd=['cargo','test','-p','ente-ensu','--lib','--no-default-features','retrieval::index::tests::benchmarks_dictionary_10k_semantic_queries','--','--nocapture']
        print('Running semantic benchmark through Ensu Rust RetrievalIndex...',flush=True)
        p=subprocess.run(cmd,cwd=args.upstream,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        print(p.stdout)
        if p.returncode: raise SystemExit(p.returncode)
        m=re.search(r'BENCH_SUMMARY\|(\d+)\|(\d+)\|(\d+)',p.stdout)
        if not m: raise SystemExit('Benchmark summary not found in Rust output')
        total,top1,top5=map(int,m.groups())
        report={'dataset':'dictionary-10k-v1','cases':total,'top1_correct':top1,'top5_correct':top5,'top1_accuracy':round(top1/total,4),'top5_accuracy':round(top5/total,4),'skipped_missing_headwords':[x['expected'] for x in skipped]}
        pathlib.Path('semantic-quality-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('QUALITY_REPORT='+json.dumps(report),flush=True)
    finally:
        src.write_text(original)

if __name__=='__main__': main()
