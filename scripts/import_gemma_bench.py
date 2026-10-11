"""Copy gemma4agentcomp's issue, idq and run-F/G agent queries and file labels.

Snapshots stay at the source path. Neither patches nor task labels enter the
word-table builder. Query extraction intentionally reuses the original helpers.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('benchmarks/gemma/queries.jsonl'))
    args = parser.parse_args()
    source = args.source.resolve()
    spec = importlib.util.spec_from_file_location('gemma_eval', source/'scripts/eval_search.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    traces = [source/'artifacts/EXP-020-run-f/full/results/exp020/traces',
              source/'artifacts/EXP-020-run-g/full/results/exp020_6min/traces']
    for path in traces:
        if not path.is_dir():
            raise FileNotFoundError(path)
    agent = module.agent_queries(traces)
    tasks_path = source/'data/competition/tasks.jsonl'
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    rows = []
    for task in tasks:
        task_id = task['instance_id']
        queries = {'issue': task['problem_statement'],
                   'idq': module.identifier_query(task['problem_statement'])}
        if task_id in agent:
            queries['agent'] = agent[task_id]
        for mode, query in queries.items():
            rows.append({'id': task_id+':'+mode, 'task': task_id, 'repo': task['repo'],
                         'base_commit': task['base_commit'], 'mode': mode, 'query': query,
                         'refs': sorted(module.ref_files(task['patch']))})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    metadata = {'source': str(source), 'tasks_sha256': hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
                'query_extractor_sha256': hashlib.sha256((source/'scripts/eval_search.py').read_bytes()).hexdigest(),
                'trace_dirs': [str(t) for t in traces], 'tasks': len(tasks),
                'queries': {mode: sum(r['mode'] == mode for r in rows) for mode in ('issue','idq','agent')}}
    args.out.with_suffix('.source.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
