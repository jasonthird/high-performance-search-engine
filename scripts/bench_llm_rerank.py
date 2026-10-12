# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["torch>=2.5", "sentence-transformers>=5.0,<5.2", "transformers>=4.51,<5", "accelerate", "mxbai-rerank"]
# ///
"""LLM-based rerankers over the stored hybrid pools of bench_rerank.py.

Reads target/rerank-bench/raw.jsonl (first-stage pools are already there), so
only the reranker runs. Each model scores the top 20 hybrid chunks of every
query and writes llm-<model>.jsonl beside raw.jsonl; report_rerank.py merges
them. Query and chunk are pre-truncated by token count (QUERY_TOKENS,
DOC_TOKENS) so prompt templates are never cut. Latency is PyTorch on MPS.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from bench_static_query import corpus  # noqa: E402

DEPTH = 20
QUERY_TOKENS = 384
DOC_TOKENS = 640
INSTRUCTION = ('Given a GitHub issue or code search query, retrieve the source code '
               'that must be read or changed to resolve it')
QWEN_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on '
               'the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
               '<|im_end|>\n<|im_start|>user\n')
QWEN_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
MODELS = {
    'qwen3-0.6b': ('tomaarsen/Qwen3-Reranker-0.6B-seq-cls', 'qwen'),
    'qwen3-4b': ('tomaarsen/Qwen3-Reranker-4B-seq-cls', 'qwen'),
    'mxbai-v2-base': ('mixedbread-ai/mxbai-rerank-base-v2', 'mxbai'),
    'jina-v3': ('jinaai/jina-reranker-v3', 'listwise'),
}


class Scorer:
    def __init__(self, repo, kind):
        import torch
        self.torch, self.kind = torch, kind
        if kind == 'mxbai':
            # Its yes/no head lives in the causal LM; CrossEncoder loads a random head.
            from mxbai_rerank import MxbaiRerankV2
            self.model = MxbaiRerankV2(repo, device='mps', torch_dtype=torch.bfloat16)
            self.tok = self.model.tokenizer
            self.params = sum(p.numel() for p in self.model.model.parameters())
            return
        if kind == 'listwise':
            from transformers import AutoModel, AutoTokenizer
            self.model = AutoModel.from_pretrained(repo, dtype=torch.bfloat16, trust_remote_code=True).to('mps').eval()
            self.tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        else:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(repo, device='mps', max_length=QUERY_TOKENS + DOC_TOKENS + 128,
                                      model_kwargs={'dtype': torch.bfloat16})
            self.tok = self.model.tokenizer
        self.params = sum(p.numel() for p in (self.model if kind == 'listwise' else self.model.model).parameters())

    def cut(self, text, n):
        ids = self.tok(text, add_special_tokens=False)['input_ids']
        return text if len(ids) <= n else self.tok.decode(ids[:n])

    def score(self, query, docs):
        query = self.cut(query, QUERY_TOKENS)
        docs = [self.cut(d, DOC_TOKENS) for d in docs]
        t = time.perf_counter()
        with self.torch.inference_mode():
            if self.kind == 'mxbai':
                by_index = {r.index: r.score for r in self.model.rank(query, docs, batch_size=10)}
                scores = [float(by_index[i]) for i in range(len(docs))]
            elif self.kind == 'listwise':
                by_index = {r['index']: r['relevance_score'] for r in self.model.rerank(query, docs)}
                scores = [float(by_index[i]) for i in range(len(docs))]
            else:
                if self.kind == 'qwen':
                    pairs = [(f'{QWEN_PREFIX}<Instruct>: {INSTRUCTION}\n<Query>: {query}\n',
                              f'<Document>: {d}{QWEN_SUFFIX}') for d in docs]
                else:
                    pairs = [(query, d) for d in docs]
                scores = [float(s) for s in self.model.predict(pairs, batch_size=10, show_progress_bar=False)]
        self.torch.mps.synchronize()
        return scores, (time.perf_counter() - t) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--results', type=Path, default=Path('target/rerank-bench'))
    parser.add_argument('--models', default=','.join(MODELS))
    args = parser.parse_args()
    raw = [json.loads(l) for l in (args.results / 'raw.jsonl').read_text().splitlines()]
    for name in args.models.split(','):
        repo, kind = MODELS[name]
        out = args.results / f'llm-{name}.jsonl'
        done = {json.loads(l)['task'] for l in out.read_text().splitlines()} if out.exists() else set()
        if len(done) == len(raw):
            continue
        t = time.perf_counter()
        scorer = Scorer(repo, kind)
        scorer.score('warm up', ['def f():\n    return 1'] * 4)
        print(f'loaded {name} ({scorer.params/1e6:.0f}M) in {time.perf_counter()-t:.0f} s', flush=True)
        meta = args.results / 'models.json'
        models = json.loads(meta.read_text())
        models[name] = {'repo': repo, 'params': scorer.params, 'device': 'mps', 'dtype': 'bfloat16'}
        meta.write_text(json.dumps(models, indent=2) + '\n')
        for i, task in enumerate(raw):
            if task['task'] in done:
                continue
            text = {d['id']: f"{d['title']}\n{d['body']}" for d in corpus(args.snapshots / (task['task'] + '.tgz'))}
            rows = []
            for q in task['results']:
                ids = q['pools']['hybrid_full'][:DEPTH]
                scores, ms = scorer.score(q['query'], [text[x] for x in ids])
                rows.append({'id': q['id'], 'rerank': {f'{name}@hybrid_full': {'scores': scores, 'ms_first20': ms, 'ms_all': ms}}})
            with out.open('a') as f:
                f.write(json.dumps({'task': task['task'], 'results': rows}) + '\n')
            print(f'{name} {i+1}/{len(raw)} {task["task"]}: median {statistics.median(r["rerank"][f"{name}@hybrid_full"]["ms_first20"] for r in rows):.0f} ms', flush=True)
        del scorer


if __name__ == '__main__':
    main()
