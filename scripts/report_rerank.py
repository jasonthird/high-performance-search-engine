"""Summarize bench_rerank.py output: first-stage modes vs cross-encoder reranks.

Every variant is cut to 20 chunks before collapsing duplicate paths, matching
the first-stage variants. R@k is the fraction of queries whose first reference
file is within k files, as in report_static_query.py.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

TOP_CHUNKS = 20
RRF_K = 60


def file_rank(hits, refs):
    paths = list(dict.fromkeys(h.rsplit(':', 1)[0] for h in hits[:TOP_CHUNKS]))
    return next((i + 1 for i, p in enumerate(paths) if p in refs), None)


def reorder(ids, scores, depth):
    depth = min(depth, len(scores))  # small pools when BM25 matches few chunks
    head = sorted(range(depth), key=lambda i: -scores[i])
    return [ids[i] for i in head] + ids[depth:]


def rrf(ids, scores, depth):
    # Reciprocal rank fusion of the first-stage order and the reranker order.
    depth = min(depth, len(scores))
    by_score = sorted(range(depth), key=lambda i: -scores[i])
    rr = {i: r + 1 for r, i in enumerate(by_score)}
    head = sorted(range(depth), key=lambda i: -(1 / (RRF_K + i + 1) + 1 / (RRF_K + rr[i])))
    return [ids[i] for i in head] + ids[depth:]


def variants(q):
    out = {k: (v['hits'], None) for k, v in q['variants'].items()}
    for key, r in q['rerank'].items():
        model, pool = key.split('@')
        ids, scores = q['pools'][pool], r['scores']
        if pool == 'bm25':
            out[f'{model} | bm25 top20'] = (reorder(ids, scores, 20), r['ms_first20'])
            continue
        out[f'{model} | hybrid top20'] = (reorder(ids, scores, 20), r['ms_first20'])
        out[f'{model} | hybrid top20 rrf'] = (rrf(ids, scores, 20), r['ms_first20'])
        if len(scores) <= 20:  # LLM rerankers score only the top 20
            continue
        out[f'{model} | hybrid top50'] = (reorder(ids, scores, len(scores)), r['ms_all'])
        out[f'{model} | hybrid top50 rrf'] = (rrf(ids, scores, len(scores)), r['ms_all'])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('target/rerank-bench'))
    parser.add_argument('--queries', type=Path, default=Path('benchmarks/gemma/queries.jsonl'))
    parser.add_argument('--out', type=Path, default=Path('benchmarks/rerank'))
    args = parser.parse_args()
    labels = {r['id']: r for line in args.queries.read_text().splitlines() if (r := json.loads(line))}
    raw = [json.loads(l) for l in (args.results / 'raw.jsonl').read_text().splitlines()]
    models = json.loads((args.results / 'models.json').read_text())
    # Merge LLM reranker scores from bench_llm_rerank.py; report only complete runs.
    extra = {}
    for path in sorted(args.results.glob('llm-*.jsonl')):
        tasks = [json.loads(l) for l in path.read_text().splitlines()]
        if len(tasks) < len(raw):
            print(f'skipping incomplete {path.name}: {len(tasks)}/{len(raw)} tasks')
            continue
        for task in tasks:
            for q in task['results']:
                extra.setdefault(q['id'], {}).update(q['rerank'])
    for task in raw:
        for q in task['results']:
            q['rerank'].update(extra.get(q['id'], {}))

    ranks = defaultdict(dict)   # (mode, variant) -> {query id: rank}
    latency = defaultdict(list)
    for task in raw:
        for q in task['results']:
            assert q['pools']['hybrid_full'][:TOP_CHUNKS] == q['variants']['hybrid_full']['hits'], q['id']
            refs = set(labels[q['id']]['refs'])
            for name, (hits, ms) in variants(q).items():
                ranks[(q['mode'], name)][q['id']] = file_rank(hits, refs)
                if ms is not None:
                    latency[name].append(ms)

    def row(rs):
        n = len(rs)
        vals = list(rs.values())
        return {'n': n, **{f'recall@{k}': sum(r is not None and r <= k for r in vals) / n for k in (1, 5, 10)},
                'mrr': sum(1 / r if r else 0 for r in vals) / n}

    def paired(rs, base):
        rr = lambda r: 1 / r if r else 0
        better = sum(rr(rs[i]) > rr(base[i]) for i in rs)
        worse = sum(rr(rs[i]) < rr(base[i]) for i in rs)
        return better, worse

    modes = sorted({m for m, _ in ranks})
    order = ['bm25', 'semantic_full', 'hybrid_full', 'rerank_full']
    order += sorted({v for _, v in ranks if v not in order}, key=lambda v: (v.split(' | ')[1], v))
    summary = {m: {v: {**row(ranks[(m, v)]), 'vs_hybrid_better_worse': paired(ranks[(m, v)], ranks[(m, 'hybrid_full')]),
                       'median_rerank_ms': statistics.median(latency[v]) if latency[v] else None}
                   for v in order if (m, v) in ranks} for m in modes}
    n_tasks = len(raw)
    lines = ['# Reranker results', '',
             f'Completed {n_tasks} of 129 task snapshots. First stage: hips CodeRankEmbed + BM25 '
             '(`hybrid_full`, weighted alpha 0.15, 200 semantic candidates, exact vectors). '
             '`rerank_full` is the legacy `--mode rerank` (cosine on BM25 candidates).', '',
             'Better/worse counts compare each query\'s reciprocal file rank with `hybrid_full`.', '']
    for m in modes:
        lines += [f'## {m} queries (n={summary[m]["hybrid_full"]["n"]})', '',
                  '| Variant | R@1 | R@5 | R@10 | MRR | Better / worse vs hybrid | Rerank ms (median) |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for v, s in summary[m].items():
            b, w = s['vs_hybrid_better_worse']
            ms = f"{s['median_rerank_ms']:.0f}" if s['median_rerank_ms'] is not None else '—'
            lines.append(f"| {v} | {s['recall@1']:.3f} | {s['recall@5']:.3f} | {s['recall@10']:.3f} | "
                         f"{s['mrr']:.3f} | {'—' if v == 'hybrid_full' else f'{b} / {w}'} | {ms} |")
        lines.append('')
    lines += ['## Models', '', '| Name | Hugging Face | Params | Device |', '| --- | --- | ---: | --- |']
    for name, m in models.items():
        lines.append(f"| {name} | `{m['repo']}` | {m['params']/1e6:.0f}M | {m['device']} {m['dtype']} |")
    lines += ['', 'Latency is PyTorch (sentence-transformers) on an Apple M3, scoring all pool chunks '
              'in one batch at max 512 tokens; it is not a ggml implementation.', '']
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'results.md').write_text('\n'.join(lines))
    (args.out / 'results.json').write_text(json.dumps({'tasks': n_tasks, 'models': models, 'summary': summary}, indent=2) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
