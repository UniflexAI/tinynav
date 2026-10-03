"""Paired flat/hierarchical replay of immutable observed snapshots; no execution."""
import argparse
import copy
import json
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen
from tool.simulator.decision_observer import planning_questions, validate_response
from tool.simulator.hierarchical_decision import FAMILIES, family_questions, member_questions


def ask(state, questions, endpoint):
    payload = {'state': copy.deepcopy(state), 'questions': questions}
    started = time.monotonic()
    with urlopen(Request(endpoint, data=json.dumps(payload).encode(),
                         headers={'Content-Type': 'application/json'}), timeout=30) as response:
        result = json.load(response)
    validate_response(result, questions)
    return {'request': payload, 'response': result,
            'latency_ms': round((time.monotonic() - started) * 1000, 1)}


def evaluate(state, kind, endpoint):
    if kind == 'flat':
        calls = [ask(state, planning_questions(state), endpoint)]
    else:
        calls = [ask(state, family_questions(state), endpoint)]
        family = calls[0]['response']['answers']['alternative']['choice']
        if family in FAMILIES:
            calls.append(ask(state, member_questions(state, family), endpoint))
    choice = calls[-1]['response']['answers']['alternative']['choice']
    allowed = planning_questions(state)['alternative']['criteria']
    if choice not in allowed:
        raise ValueError('Final choice is not an existing strategy or abstention')
    return {'calls': calls, 'choice': choice,
            'latency_ms': round(sum(call['latency_ms'] for call in calls), 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8090/v1/systemone')
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError('repeats must be positive')
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    sources = sorted(Path(args.snapshots).glob('snapshot-*.json'))
    if not sources:
        raise ValueError('No snapshots')
    for repeat in range(args.repeats):
        for index, source_path in enumerate(sources):
            source = json.loads(source_path.read_text())
            state = source['evaluations']['v2']['request']['state']
            frozen = json.dumps(state, sort_keys=True)
            order = ['flat', 'hierarchical'] if (index + repeat) % 2 == 0 else ['hierarchical', 'flat']
            row = {'source_id': source['source_id'], 'scenario': source['scenario'],
                   'source_file': str(source_path), 'repeat': repeat + 1, 'order': order, 'evaluations': {}}
            for kind in order:
                row['evaluations'][kind] = evaluate(state, kind, args.endpoint)
            if json.dumps(state, sort_keys=True) != frozen:
                raise ValueError('Source mutated')
            row['choices'] = {kind: result['choice'] for kind, result in row['evaluations'].items()}
            rows.append(row)
            (folder / ('pair-' + str(len(rows)) + '.json')).write_text(json.dumps(row, indent=2))
            print(row['scenario'], row['repeat'], row['choices'], flush=True)
    summary = {'kind': 'same_observed_state_flat_vs_two_stage_questions_no_execution',
               'snapshots': len(sources), 'pairs': len(rows), 'repeats': args.repeats,
               'changed': sum(row['choices']['flat'] != row['choices']['hierarchical'] for row in rows),
               'groups': {}}
    for scenario in sorted({row['scenario'] for row in rows}):
        group = [row for row in rows if row['scenario'] == scenario]
        summary['groups'][scenario] = {kind: {
            'choices': [row['choices'][kind] for row in group],
            'median_latency_ms': statistics.median(row['evaluations'][kind]['latency_ms'] for row in group),
            'max_latency_ms': max(row['evaluations'][kind]['latency_ms'] for row in group),
            'over_existing_8s_ttl': sum(row['evaluations'][kind]['latency_ms'] > 8000 for row in group),
            'calls': sum(len(row['evaluations'][kind]['calls']) for row in group)}
            for kind in ('flat', 'hierarchical')}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
