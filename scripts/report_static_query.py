"""Publish a compact, reviewable report from a completed paired benchmark."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--queries', type=Path, default=Path('benchmarks/gemma/queries.jsonl'))
    parser.add_argument('--words', type=Path, default=Path('data/static-query/words-768.json'))
    parser.add_argument('--out', type=Path, default=Path('benchmarks/gemma/results'))
    args = parser.parse_args()
    report = json.loads((args.results / 'report.json').read_text())
    if report['completed_tasks'] != report['requested_tasks']:
        raise ValueError('benchmark is incomplete; refusing to publish partial results')
    if hashlib.sha256(args.words.read_bytes()).hexdigest() != report['identity']['word_table_sha256']:
        raise ValueError('word table does not match benchmark inputs')
    report['word_table'] = {'path': str(args.words), 'bytes': args.words.stat().st_size}
    raw = [json.loads(line) for line in (args.results / 'raw.jsonl').read_text().splitlines()]
    labels = {r['id']: r for line in args.queries.read_text().splitlines() if (r := json.loads(line))}
    if len({row['task'] for row in raw}) != len(raw):
        raise ValueError('duplicate task results')
    spec = importlib.util.spec_from_file_location('bench', Path(__file__).with_name('bench_static_query.py'))
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    summary = bench.summarize(raw, labels)
    if summary != report['summary']:
        raise ValueError('saved summary does not match raw results')
    by_repo = defaultdict(list)
    for task in raw:
        by_repo[labels[task['results'][0]['id']]['repo']].append(task)
    report['by_repo'] = {repo: bench.summarize(rows, labels) for repo, rows in sorted(by_repo.items())}
    report['runtime'] = {
        'encoder': raw[0]['encoder'],
        'static_load_ms': raw[0]['static_load_ms'],
        'full_document_and_query_load_ms': raw[0]['full_load_ms'],
        'distinct_document_chunks_encoded': sum(row['encoded'] for row in raw),
    }
    report['static_query_errors'] = [
        {'task': task['task'], 'id': q['id'], 'query': q['query'], 'error': q['static_error']}
        for task in raw for q in task['results'] if q['static_error']
    ]
    paired = defaultdict(Counter)
    growing_paired = defaultdict(Counter)
    for task in raw:
        for q in task['results']:
            def top10_hit(variant):
                hits = q['variants'].get(variant, {}).get('hits', [])
                paths = list(dict.fromkeys(hit.rsplit(':', 1)[0] for hit in hits))
                return any(path in labels[q['id']]['refs'] for path in paths[:10])
            full, static = top10_hit('hybrid_full'), top10_hit('hybrid_static')
            category = 'both' if full and static else 'full_only' if full else 'static_only' if static else 'neither'
            paired[q['mode']][category] += 1
            if q.get('dynamic'):
                growing_hit = top10_hit('hybrid_dynamic')
                category = 'both' if full and growing_hit else 'full_only' if full else 'lookup_only' if growing_hit else 'neither'
                growing_paired[q['mode']][category] += 1
    report['paired_hybrid_top10'] = {mode: dict(counts) for mode, counts in sorted(paired.items())}
    lines = ['# Persistent word lookup results' if report['identity'].get('growing_word_cache') else '# 768-dimensional static query results', '',
        f"Completed {len(raw)} task snapshots. Each static/full pair scores identical full-model document vectors.", '',
        f"Word table: `{args.words}`, {args.words.stat().st_size / 1_000_000:.1f} MB.", '',
        '| Queries | Encoder / ranking | N | R@1 | R@10 | MRR | Encode ms (median; full cache hits excluded) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for mode in ('agent', 'idq', 'issue'):
        for variant in bench.variant_names(raw):
            m = summary[mode][variant]
            ms = m['median_uncached_encode_ms']
            lines.append(f"| {mode} | {variant} | {m['n']} | {m['recall@1']:.3f} | {m['recall@10']:.3f} | {m['mrr']:.3f} | {ms:.4f} |" if ms is not None else
                         f"| {mode} | {variant} | {m['n']} | {m['recall@1']:.3f} | {m['recall@10']:.3f} | {m['mrr']:.3f} | — |")
    errors = sum(m['errors'] for variants in summary.values() for m in variants.values())
    coverages = [q['coverage'] for task in raw for q in task['results'] if q['coverage'] is not None]
    lines += ['', f"Variant errors: {errors}. Median vocabulary coverage: {statistics.median(coverages):.1%}.", '',
        f"Backend: `{raw[0]['encoder']['backend']}`, device `{raw[0]['encoder']['ggml_device']}`.",
        f"Word-table load: {raw[0]['static_load_ms']:.1f} ms; both full encoders loaded in {raw[0]['full_load_ms']:.1f} ms.", '',
        'Latency measures a resident process. It excludes full-query cache hits and does not promise lower cold-start latency.',
        'The static table is an approximation; JSON lookups do not reproduce contextual model predictions.', '',
        'See [methodology and commands](README.md). Full precision metrics, per-repository results and input hashes are in the adjacent JSON.', '',
        f'Raw outputs: `{args.results}/raw.jsonl` (generated, ignored by Git).', '']
    if report['static_query_errors']:
        lines += ['## Queries requiring fallback', '']
        for error in report['static_query_errors']:
            lines.append(f"- `{error['id']}`: `{error['query']}` — {error['error']}.")
        lines += ['', 'These count as misses in the measured static variants. The search command supports `--fallback-full`.', '']
    lines += ['## Paired hybrid top-10 outcomes', '',
              '| Queries | Both succeed | Full only | Static only | Neither |',
              '| --- | ---: | ---: | ---: | ---: |']
    for mode in ('agent', 'idq', 'issue'):
        counts = paired[mode]
        lines.append(f"| {mode} | {counts['both']} | {counts['full_only']} | {counts['static_only']} | {counts['neither']} |")
    lines += ['', 'Equal aggregate success rates can hide different query outcomes; compare ranking quality as well.', '']
    dynamic = [q for t in raw for q in t['results'] if q.get('dynamic')]
    if dynamic:
        cache_path = args.results / 'learned-words.json'
        stats = {'queries':len(dynamic),
                 'queries_with_misses':sum(q['dynamic']['missing_terms'] > 0 for q in dynamic),
                 'distinct_words_added':sum(q['dynamic']['missing_terms'] for q in dynamic),
                 'sidecar_bytes':cache_path.stat().st_size,
                 'repeat_model_calls':0,
                 'median_repeat_encode_ms':statistics.median(q['dynamic']['repeat_encode_ms'] for q in dynamic),
                 'median_first_miss_encode_ms':statistics.median(q['dynamic']['encode_ms'] for q in dynamic if q['dynamic']['missing_terms']),
                 'by_mode':{mode:{'queries':sum(q['mode']==mode for q in dynamic),
                                  'queries_with_misses':sum(q['mode']==mode and q['dynamic']['missing_terms']>0 for q in dynamic),
                                  'median_first_encode_ms':statistics.median(q['dynamic']['encode_ms'] for q in dynamic if q['mode']==mode),
                                  'median_repeat_encode_ms':statistics.median(q['dynamic']['repeat_encode_ms'] for q in dynamic if q['mode']==mode)} for mode in ('agent','idq','issue')}}
        if len(json.loads(cache_path.read_text())['vectors']) != stats['distinct_words_added']:
            raise ValueError('saved word cache does not match measured misses')
        stats['paired_hybrid_top10'] = {mode:dict(counts) for mode, counts in sorted(growing_paired.items())}
        report['growing_lookup'] = stats
        lines += ['## Persistent word lookup', '',
                  f"Added {stats['distinct_words_added']} distinct words across {stats['queries_with_misses']}/{stats['queries']} first-pass queries; sidecar {stats['sidecar_bytes']/1_000_000:.2f} MB.",
                  f"All {stats['queries']} immediate repeats reproduced the same vector with zero model calls. Median repeat encoding: {stats['median_repeat_encode_ms']:.4f} ms.",
                  f"Median first-pass encoding for queries with misses: {stats['median_first_miss_encode_ms']:.3f} ms, including atomic JSON persistence.", '',
                  'Misses are encoded as individual words using the base table’s prefix setting, then quantized to the same 768-dimensional format. The full query is still an approximation.',
                  'This sidecar assumes a single writer. Document vectors are reused from the original experiment; query timing comparisons are measured again in this run.',
                  'Growing first-pass timings include existing word-cache hits from earlier benchmark queries. Issue queries are submitted before identifier and agent queries for each task.', '',
                  '| Queries | With word misses | First-pass encode ms | Repeated encode ms |',
                  '| --- | ---: | ---: | ---: |']
        for mode, m in stats['by_mode'].items():
            lines.append(f"| {mode} | {m['queries_with_misses']}/{m['queries']} | {m['median_first_encode_ms']:.4f} | {m['median_repeat_encode_ms']:.4f} |")
        lines += ['', '| Queries | Full and lookup succeed | Full only | Lookup only | Neither |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for mode in ('agent','idq','issue'):
            counts = growing_paired[mode]
            lines.append(f"| {mode} | {counts['both']} | {counts['full_only']} | {counts['lookup_only']} | {counts['neither']} |")
        lines += ['']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    args.out.with_suffix('.md').write_text('\n'.join(lines))
    print(args.out.with_suffix('.md'))


if __name__ == '__main__':
    main()
