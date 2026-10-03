"""Separate factual-reading and alternative-option-order controls, no execution."""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from tool.simulator.directional_information import explicit, mirror, compact_state_text, swap_direction
from tool.simulator.decision_observer import planning_questions

READING = {'type': 'choice', 'instructions':
    'Read planning.representatives.left and right. Which side has the lower collision_rejected / count fraction? '
    'Compare the reported numbers only; this is not a claim of measured footprint safety or escape success.',
    'criteria': {'left': 'Left has a strictly lower rejection fraction',
                 'right': 'Right has a strictly lower rejection fraction',
                 'same': 'Both fractions are equal', 'missing': 'At least one count is missing or zero'}}


def expected_reading(state):
    representatives = state['planning']['representatives']
    left, right = representatives.get('left', {}), representatives.get('right', {})
    if not left.get('count') or not right.get('count') or 'collision_rejected' not in left or 'collision_rejected' not in right:
        return 'missing'
    a, b = left['collision_rejected'] / left['count'], right['collision_rejected'] / right['count']
    return 'same' if a == b else 'left' if a < b else 'right'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshots', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--runtime-source', required=True)
    p.add_argument('--model-dir', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.runtime_source)
    from transformers import AutoTokenizer
    from startlux_decision import jevfmt as J
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    with urlopen('http://127.0.0.1:8081/props') as response:
        capacity = json.load(response)['default_generation_settings']['n_ctx']
    folder = Path(a.output)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(sorted(Path(a.snapshots).glob('snapshot-*.json'))):
        source = json.loads(path.read_text())
        original = source['evaluations']['v2']['request']['state']
        row = {'source_id': source['source_id'], 'scenario': source['scenario'], 'evaluations': {}}
        geometries = [('original', original), ('mirror', mirror(original))]
        if index % 2:
            geometries.reverse()
        for geometry, raw in geometries:
            named = explicit(raw)
            evidence = compact_state_text(named)
            spec = planning_questions(named)['alternative']
            orders = ('normal', 'reversed') if index % 2 == 0 else ('reversed', 'normal')
            for order in orders:
                alternative = dict(spec)
                alternative['criteria'] = spec['criteria'] if order == 'normal' else dict(reversed(list(spec['criteria'].items())))
                questions = {'alternative': alternative, 'reading': READING}
                audit = {}
                for key, question in questions.items():
                    rendered = J.from_systemone(evidence, question, key)
                    ids, _ = J.render_ids(rendered, tok, max_length=capacity - 1)
                    msgs, letters = J.messages(rendered)
                    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    file = path.stem + '-' + geometry + '-' + order + '-' + key + '.txt'
                    (folder / file).write_text(text)
                    audit[key] = {'tokens': len(ids), 'file': file, 'option_order': letters}
                payload = {'state': evidence, 'questions': questions}
                started = time.monotonic()
                with urlopen(Request('http://127.0.0.1:8090/v1/systemone', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'}), timeout=30) as response:
                    result = json.load(response)
                latency = round((time.monotonic() - started) * 1000, 1)
                for key, question in questions.items():
                    answer = result['answers'][key]
                    probs = answer['probabilities']
                    if set(probs) != set(question['criteria']) or answer['choice'] not in probs:
                        raise ValueError('Unexpected answer options')
                    if not all(math.isfinite(v) and 0 <= v <= 1 for v in probs.values()) or abs(sum(probs.values()) - 1) > .001:
                        raise ValueError('Invalid probabilities')
                if result['usage']['input_tokens'] != sum(q['tokens'] for q in audit.values()):
                    raise ValueError('Runtime differs from preflight')
                record = {'request': payload, 'response': result, 'latency_ms': latency,
                          'prompt_audit': audit, 'expected_reading': expected_reading(raw)}
                row['evaluations'][geometry + '-' + order] = record
                print(source['scenario'], geometry, order, result['answers']['alternative']['choice'],
                      result['answers']['reading']['choice'], 'expected', record['expected_reading'], flush=True)
        rows.append(row)
        (folder / ('probe-' + str(len(rows)) + '.json')).write_text(json.dumps(row, indent=2))
    if not rows:
        raise ValueError('No snapshots')
    summary = {'kind': 'offline_factual_reading_and_option_order_diagnostic', 'snapshots': len(rows),
               'requests': 4 * len(rows), 'groups': {}}
    for scenario in sorted({r['scenario'] for r in rows}):
        group = [r for r in rows if r['scenario'] == scenario]
        evaluations = [e for r in group for e in r['evaluations'].values()]
        summary['groups'][scenario] = {
            'reading_correct': sum(e['response']['answers']['reading']['choice'] == e['expected_reading'] for e in evaluations),
            'reading_checks': len(evaluations),
            'order_changed': sum(r['evaluations'][geometry + '-normal']['response']['answers']['alternative']['choice'] !=
                                 r['evaluations'][geometry + '-reversed']['response']['answers']['alternative']['choice']
                                 for r in group for geometry in ('original', 'mirror')),
            'order_pairs': 2 * len(group),
            'choices': [{name: e['response']['answers']['alternative']['choice'] for name, e in r['evaluations'].items()} for r in group],
            'median_latency_ms': statistics.median(e['latency_ms'] for e in evaluations)}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
