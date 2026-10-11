"""Paired file-retrieval benchmark on gemma4agentcomp's exact task snapshots.

Uses the same code-only 80-line windows / 60-line stride and issue/idq/agent
queries as gemma's benchmark. Rust uses hips BM25/fusion and CLS document
embeddings; full and static queries score identical FP16 document vectors.
No reference labels are sent to the search process.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import statistics
import subprocess
import tarfile

EXTENSIONS = {'.py','.pyi','.js','.jsx','.ts','.tsx','.rs','.go','.java',
              '.c','.h','.cpp','.hpp','.sh','.rb','.cs'}
SKIP = {'.git','.venv','venv','node_modules','__pycache__','dist','build','.tox'}
VARIANTS = ('bm25','semantic_full','semantic_static','hybrid_full','hybrid_static')


def corpus(snapshot):
    files = {}
    with tarfile.open(snapshot, 'r|gz') as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if (not member.isfile() or member.size > 512_000 or path.is_absolute()
                    or '..' in path.parts or path.suffix.lower() not in EXTENSIONS
                    or any(p in SKIP or p.startswith('.') for p in path.parts)
                    or path.name.startswith('.adk_exec_')):
                continue
            raw = archive.extractfile(member).read()
            if b'\0' in raw:
                continue
            try:
                files[path.as_posix()] = raw.decode('utf-8')
            except UnicodeError:
                continue
    docs = []
    used = 0
    # Match os.walk's sorted directory traversal: a directory's files precede
    # its child directories, including root files before any subtree.
    for path, text in sorted(files.items(), key=lambda pair: (PurePosixPath(pair[0]).parts[:-1], PurePosixPath(pair[0]).name)):
        if used >= 32_000_000 or len(docs) >= 20000:
            raise ValueError('corpus exceeded benchmark limits; refusing a partial index')
        used += len(text.encode('utf-8'))
        lines = text.splitlines()
        for start in range(0, len(lines), 60):
            end = min(start+80,len(lines))
            body = '\n'.join(lines[start:end])
            docs.append({'id':f'{path}:{start+1}-{end}','title':path,'body':body})
            if end == len(lines):
                break
    if not docs:
        raise ValueError('empty snapshot corpus')
    return docs


def variant_names(raw):
    return VARIANTS + (('semantic_dynamic','hybrid_dynamic') if any('hybrid_dynamic' in q['variants'] for t in raw for q in t['results']) else ())


def summarize(raw, labels):
    grouped = defaultdict(list)
    for task in raw:
        for result in task['results']:
            label = labels[result['id']]
            for variant in variant_names(raw):
                value = result['variants'].get(variant)
                paths = list(dict.fromkeys(h.rsplit(':',1)[0] for h in value['hits'])) if value else []
                rank = next((i+1 for i,p in enumerate(paths) if p in label['refs']),None)
                grouped[(label['mode'],variant)].append((rank,result,value))
    summary = {}
    for (mode,variant), rows in sorted(grouped.items()):
        n = len(rows)
        timings = [v['encode_ms'] for _,r,v in rows if v and 'encode_ms' in v and
                   (not variant.endswith('full') or not r['full_cached'])]
        summary.setdefault(mode,{})[variant] = {'n':n,
            **{f'recall@{k}':sum(rank is not None and rank <= k for rank,_,_ in rows)/n for k in (1,3,5,10,20)},
            'mrr':sum(1/rank if rank else 0 for rank,_,_ in rows)/n,
            'errors':sum(v is None for _,_,v in rows),
            'median_uncached_encode_ms':statistics.median(timings) if timings else None,
            'median_total_ms':statistics.median(v['total_ms'] for _,_,v in rows if v) if any(v for _,_,v in rows) else None}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--queries', type=Path, default=Path('benchmarks/gemma/queries.jsonl'))
    parser.add_argument('--words', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path('target/release/examples/static_query'))
    parser.add_argument('--out', type=Path, default=Path('target/static-query-bench'))
    parser.add_argument('--limit', type=int, default=0, help='zero runs all tasks; otherwise balanced repo sample')
    parser.add_argument("--grow", action="store_true", help="encode and persist missing words")
    args = parser.parse_args()
    labels = {r['id']:r for line in args.queries.read_text().splitlines() if (r:=json.loads(line))}
    jobs = defaultdict(list)
    for row in labels.values():
        jobs[row['task']].append(row)
    # Round robin between repositories so even a short smoke run covers all four.
    by_repo = defaultdict(list)
    for task,queries in jobs.items():
        by_repo[queries[0]['repo']].append(task)
    ordered = []
    while any(by_repo.values()):
        for repo in sorted(by_repo):
            if by_repo[repo]:
                ordered.append(by_repo[repo].pop(0))
    if args.limit:
        ordered = ordered[:args.limit]
    args.out.mkdir(parents=True,exist_ok=True)
    sha = hashlib.sha256(args.words.read_bytes()).hexdigest()
    identity = {'word_table_sha256':sha,'queries_sha256':hashlib.sha256(args.queries.read_bytes()).hexdigest(),
                'binary_sha256':hashlib.sha256(args.binary.read_bytes()).hexdigest()}
    if args.grow:
        identity['growing_word_cache'] = True
    identity_path = args.out/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('benchmark inputs changed; use a new --out directory')
    identity_path.write_text(json.dumps(identity,indent=2)+'\n')
    raw_path = args.out/'raw.jsonl'
    raw = [json.loads(line) for line in raw_path.read_text().splitlines()] if raw_path.exists() else []
    done = {r['task'] for r in raw}
    with (args.out/'encoder.log').open('a') as log:
        command = [str(args.binary.resolve()),'bench','--words',str(args.words.resolve()),
            '--cache',str((args.out/'doc-cache').resolve())]
        if args.grow:
            command += ['--grow-cache',str((args.out/'learned-words.json').resolve())]
        process = subprocess.Popen(command, stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=log,text=True,bufsize=1)
        try:
            for task in ordered:
                if task in done:
                    continue
                snapshot = args.snapshots/(task+'.tgz')
                docs = corpus(snapshot)
                corpus_path = args.out/'corpus.jsonl'
                corpus_path.write_text(''.join(json.dumps(d)+'\n' for d in docs))
                # The original tool bounds raw issues to 8,000 characters.
                queries = [{'id':r['id'],'mode':r['mode'],'query':r['query'][:8000]} for r in jobs[task]]
                process.stdin.write(json.dumps({'task':task,'corpus':str(corpus_path.resolve()),'queries':queries})+'\n')
                process.stdin.flush()
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError(f'benchmark worker stopped on {task}; inspect {args.out}/encoder.log')
                result = json.loads(line)
                result['corpus_sha256'] = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
                result['snapshot'] = {'name':snapshot.name,'bytes':snapshot.stat().st_size,
                                      'mtime_ns':snapshot.stat().st_mtime_ns}
                raw.append(result)
                with raw_path.open('a') as output:
                    output.write(json.dumps(result)+'\n')
                report = {'identity':identity,'completed_tasks':len(raw),'requested_tasks':len(ordered),
                          'settings':{'chunk_window':80,'chunk_stride':60,'top_chunks':20,'semantic_candidates':200,
                                      'fusion':'weighted','alpha':0.15,'exact_vectors':True},
                          'summary':summarize(raw,labels)}
                (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
                print(f'{len(raw)}/{len(ordered)} {task}: {len(docs)} chunks, {result["encoded"]} encoded',flush=True)
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
    print(json.dumps(summarize(raw,labels),indent=2))


if __name__ == '__main__':
    main()
