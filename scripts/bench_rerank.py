# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["torch>=2.5", "sentence-transformers>=4.1", "transformers>=4.48,<5", "einops"]
# ///
"""Second-stage reranker benchmark on gemma4agentcomp's task snapshots.

Stage 1 is the Rust worker of bench_static_query.py (same corpus, chunking,
queries and labels): it reports BM25, semantic, hybrid and the legacy
BM25-then-cosine `rerank` mode, plus 50-chunk BM25 and hybrid pools. Stage 2
scores every pool chunk with small cross-encoders and stores the raw scores,
so pool depth and fusion are evaluated offline by report_rerank.py.

Reranker latency is PyTorch on this machine, not a ggml implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from bench_static_query import corpus  # noqa: E402

# Small cross-encoders only (<300M parameters); see benchmarks/rerank/README.md.
# Each runs on its fastest measured PyTorch device: DeBERTa is slower on MPS
# than on the CPU, the others are fastest on MPS in fp16.
MODELS = {
    'minilm-l6': ('cross-encoder/ms-marco-MiniLM-L6-v2', {}, 'mps'),
    'jina-v1-turbo': ('jinaai/jina-reranker-v1-turbo-en', {'trust_remote_code': True}, 'mps'),
    'mxbai-xsmall': ('mixedbread-ai/mxbai-rerank-xsmall-v1', {}, 'cpu'),
    'mxbai-base': ('mixedbread-ai/mxbai-rerank-base-v1', {}, 'cpu'),
    'gte-modernbert': ('Alibaba-NLP/gte-reranker-modernbert-base', {}, 'mps'),
    'jina-v2-base': ('jinaai/jina-reranker-v2-base-multilingual', {'trust_remote_code': True}, 'mps'),
}
# Chunks scored per first-stage pool; the BM25 pool is a cheaper secondary check.
DEPTH = {'hybrid_full': 50, 'bm25': 20}
MAX_LENGTH = 512
FIRST = 20  # timed separately: latency for a 20-chunk pool


def load_models(names):
    import torch
    from sentence_transformers import CrossEncoder
    models = {}
    for name in names:
        repo, kwargs, device = MODELS[name]
        dtype = torch.float16 if device == 'mps' else torch.float32
        t = time.perf_counter()
        model = CrossEncoder(repo, max_length=MAX_LENGTH, device=device,
                             model_kwargs={'torch_dtype': dtype}, **kwargs)
        model.model.eval()
        params = sum(p.numel() for p in model.model.parameters())
        # Warm kernels so the first query is not charged for compilation.
        model.predict([('warm up', 'def f():\n    return 1')] * 8)
        models[name] = {'model': model, 'repo': repo, 'params': params, 'device': device,
                        'dtype': str(dtype).removeprefix('torch.'),
                        'load_ms': (time.perf_counter() - t) * 1000}
        print(f'loaded {name} ({params/1e6:.0f}M) on {device}', flush=True)
    return models, torch


def score(entry, torch, query, texts):
    def run(batch):
        if not batch:
            return [], 0.0
        t = time.perf_counter()
        out = entry['model'].predict([(query, x) for x in batch], batch_size=len(batch),
                                     convert_to_numpy=True, show_progress_bar=False)
        if entry['device'] == 'mps':
            torch.mps.synchronize()
        return [float(s) for s in out], (time.perf_counter() - t) * 1000
    head, head_ms = run(texts[:FIRST])
    tail, tail_ms = run(texts[FIRST:])
    return head + tail, head_ms, head_ms + tail_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--queries', type=Path, default=Path('benchmarks/gemma/queries.jsonl'))
    parser.add_argument('--words', type=Path, required=True, help='static word table the worker loads')
    parser.add_argument('--binary', type=Path, default=Path('target/release/examples/static_query'))
    parser.add_argument('--out', type=Path, default=Path('target/rerank-bench'))
    parser.add_argument('--models', default=','.join(MODELS))
    parser.add_argument('--limit', type=int, default=0, help='zero runs all tasks; otherwise balanced repo sample')
    args = parser.parse_args()
    names = [m for m in args.models.split(',') if m]
    unknown = set(names) - set(MODELS)
    if unknown:
        parser.error(f'unknown models: {sorted(unknown)}')
    labels = {r['id']: r for line in args.queries.read_text().splitlines() if (r := json.loads(line))}
    jobs = {}
    for row in labels.values():
        jobs.setdefault(row['task'], []).append(row)
    by_repo = {}
    for task, queries in jobs.items():
        by_repo.setdefault(queries[0]['repo'], []).append(task)
    ordered = []
    while any(by_repo.values()):
        for repo in sorted(by_repo):
            if by_repo[repo]:
                ordered.append(by_repo[repo].pop(0))
    if args.limit:
        ordered = ordered[:args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    identity = {'queries_sha256': hashlib.sha256(args.queries.read_bytes()).hexdigest(),
                'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                'models': {n: MODELS[n][0] + '@' + MODELS[n][2] for n in names},
                'max_length': MAX_LENGTH, 'depth': DEPTH}
    identity_path = args.out / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('benchmark inputs changed; use a new --out directory')
    identity_path.write_text(json.dumps(identity, indent=2) + '\n')
    raw_path = args.out / 'raw.jsonl'
    done = {json.loads(l)['task'] for l in raw_path.read_text().splitlines()} if raw_path.exists() else set()

    models, torch = load_models(names)
    (args.out / 'models.json').write_text(json.dumps(
        {n: {k: v for k, v in m.items() if k != 'model'} for n, m in models.items()}, indent=2) + '\n')
    with (args.out / 'encoder.log').open('a') as log:
        command = [str(args.binary.resolve()), 'bench', '--words', str(args.words.resolve()),
                   '--cache', str((args.out / 'doc-cache').resolve())]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=log, text=True, bufsize=1)
        try:
            for task in ordered:
                if task in done:
                    continue
                docs = corpus(args.snapshots / (task + '.tgz'))
                text = {d['id']: f"{d['title']}\n{d['body']}" for d in docs}
                corpus_path = args.out / 'corpus.jsonl'
                corpus_path.write_text(''.join(json.dumps(d) + '\n' for d in docs))
                queries = [{'id': r['id'], 'mode': r['mode'], 'query': r['query'][:8000]} for r in jobs[task]]
                process.stdin.write(json.dumps({'task': task, 'corpus': str(corpus_path.resolve()),
                                                'queries': queries}) + '\n')
                process.stdin.flush()
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError(f'benchmark worker stopped on {task}; inspect {args.out}/encoder.log')
                result = json.loads(line)
                for q in result['results']:
                    q['rerank'] = {}
                    for pool_name, ids in q['pools'].items():
                        if pool_name not in DEPTH:
                            continue
                        texts = [text[i] for i in ids[:DEPTH[pool_name]]]
                        for name, entry in models.items():
                            scores, ms20, ms_all = score(entry, torch, q['query'], texts)
                            q['rerank'][f'{name}@{pool_name}'] = {'scores': scores, 'ms_first20': ms20,
                                                                  'ms_all': ms_all}
                    del q['dynamic'], q['static_error']
                    q['variants'] = {k: v for k, v in q['variants'].items()
                                     if k in ('bm25', 'semantic_full', 'hybrid_full', 'rerank_full')}
                with raw_path.open('a') as output:
                    output.write(json.dumps(result) + '\n')
                done.add(task)
                ms = [r['ms_first20'] for q in result['results'] for r in q['rerank'].values()]
                print(f'{len(done)}/{len(ordered)} {task}: {len(docs)} chunks, '
                      f'median 20-chunk rerank {statistics.median(ms):.0f} ms', flush=True)
        finally:
            process.stdin.close()
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait()
                raise RuntimeError('benchmark worker did not stop')
            if code != 0:
                raise RuntimeError(f'benchmark worker failed ({code}); inspect encoder.log')


if __name__ == '__main__':
    main()
