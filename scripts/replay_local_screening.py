"""Local screening plus model-only usefulness permutations, no execution."""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from tool.simulator.directional_information import explicit, mirror, compact_state_text, swap_direction
from tool.simulator.local_candidate_screening import screen, selection
from tool.simulator.independent_candidate_evaluation import (
    USEFULNESS, usefulness_question, candidate_state, observation_grade)


def permutations(keys):
    keys = list(keys)
    return {'normal': keys, 'reverse': keys[::-1], 'rotate': keys[1:] + keys[:1]}


def validate_result(result, questions, token_count):
    for key, spec in questions.items():
        answer = result['answers'][key]
        probs = answer['probabilities']
        if set(probs) != set(spec['criteria']) or answer['choice'] not in probs:
            raise ValueError('Invalid answer options')
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1 for v in probs.values()):
            raise ValueError('Invalid probability')
        if abs(sum(probs.values()) - 1) > .001:
            raise ValueError('Invalid probability total')
    if result['usage']['input_tokens'] != token_count:
        raise ValueError('Runtime differs from audited rendering')


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
    J.check_tokenizer(tok)
    with urlopen('http://127.0.0.1:8081/props') as response:
        capacity = json.load(response)['default_generation_settings']['n_ctx']
    folder = Path(a.output)
    folder.mkdir(parents=True, exist_ok=True)
    prompts = folder / 'prompts'
    prompts.mkdir(exist_ok=True)
    utility_orders = permutations(USEFULNESS)
    grade_orders = utility_orders
    cases = []
    for path in sorted(Path(a.snapshots).glob('snapshot-*.json')):
        source = json.loads(path.read_text())
        original = source['evaluations']['v2']['request']['state']
        for geometry, state in (('original', original), ('mirror', mirror(original))):
            named = explicit(state)
            plans = named['recovery']['strategies']
            screening = screen(plans)
            if not screening['within_limit']:
                raise ValueError('Too many eligible candidates; never truncate arbitrarily')
            case = {'source_id': source['source_id'], 'source_file': str(path), 'scenario': source['scenario'],
                    'geometry': geometry, 'strategies': plans, 'screening': screening, 'candidates': {}}
            for plan in plans:
                if plan['id'] not in screening['eligible']:
                    continue
                focused = candidate_state(named, plan['id'])
                if focused['recovery']['strategies'][0] != plan:
                    raise ValueError('Candidate altered')
                evidence = compact_state_text(focused)
                views = {}
                for order in grade_orders:
                    questions = {'probe_usefulness': usefulness_question(utility_orders[order])}
                    audit = {}
                    for key, spec in questions.items():
                        row = J.from_systemone(evidence, spec, key)
                        ids, _ = J.render_ids(row, tok, max_length=capacity - 1)
                        msgs, options = J.messages(row)
                        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                        with urlopen(Request('http://127.0.0.1:8081/tokenize', data=json.dumps(
                                {'content': text, 'add_special': False}).encode(), headers={'Content-Type': 'application/json'})) as response:
                            server_ids = json.load(response)['tokens']
                        if ids != server_ids:
                            raise ValueError('Tokenizer mismatch')
                        name = path.stem + '-' + geometry + '-' + plan['id'] + '-' + order + '-' + key + '.txt'
                        (prompts / name).write_text(text)
                        audit[key] = {'tokens': len(ids), 'option_order': options, 'file': name, 'tokenizer_parity': True}
                    views[order] = {'request': {'state': evidence, 'questions': questions}, 'prompt_audit': audit}
                case['candidates'][plan['id']] = {'reference_grade': observation_grade(plan), 'views': views}
            cases.append(case)
    if not cases:
        raise ValueError('No snapshots')
    (folder / 'preflight.json').write_text(json.dumps({'capacity': capacity, 'cases': cases}, indent=2))
    print('PREFLIGHT', len(cases), 'cases', sum(len(c['candidates']) * 3 for c in cases), 'requests, all token IDs match and prompts fit', flush=True)
    rows = []
    for index, case in enumerate(cases):
        identities = list(case['candidates'])
        candidate_orders = {'normal': identities, 'reverse': identities[::-1],
                            'rotate': identities[1:] + identities[:1]}
        order_names = list(grade_orders)
        shift = index % len(order_names)
        order_names = order_names[shift:] + order_names[:shift]
        row = {key: case[key] for key in ('source_id', 'source_file', 'scenario', 'geometry', 'screening')}
        row.update(order_sequence=order_names, evaluations={}, grades={}, usefulness={})
        started_case = time.monotonic()
        raw = json.loads(Path(case['source_file']).read_text())['evaluations']['v2']['request']['state']
        if case['geometry'] == 'mirror':
            raw = mirror(raw)
        prepared_state = explicit(raw)
        if screen(prepared_state['recovery']['strategies']) != case['screening']:
            raise ValueError('Local screening changed')
        prepared = {}
        for identity in identities:
            evidence = compact_state_text(candidate_state(prepared_state, identity))
            prepared[identity] = {}
            for order in grade_orders:
                questions = {'probe_usefulness': usefulness_question(utility_orders[order])}
                payload = {'state': evidence, 'questions': questions}
                if payload != case['candidates'][identity]['views'][order]['request']:
                    raise ValueError('Prepared request differs from preflight')
                for spec in questions.values():
                    J.render_ids(J.from_systemone(evidence, spec), tok, max_length=capacity - 1)
                prepared[identity][order] = payload
        row['local_preparation_ms'] = round((time.monotonic() - started_case) * 1000, 2)
        for order in order_names:
            records = {}
            started_order = time.monotonic()
            for identity in candidate_orders[order]:
                view = case['candidates'][identity]['views'][order]
                payload = prepared[identity][order]
                start = time.monotonic()
                with urlopen(Request('http://127.0.0.1:8090/v1/systemone', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'}), timeout=30) as response:
                    result = json.load(response)
                latency = round((time.monotonic() - start) * 1000, 1)
                validate_result(result, payload['questions'], sum(q['tokens'] for q in view['prompt_audit'].values()))
                record = {'request': payload, 'response': result, 'prompt_audit': view['prompt_audit'],
                          'latency_ms': latency, 'reference_grade': case['candidates'][identity]['reference_grade']}
                records[identity] = record
            row['evaluations'][order] = {'candidate_order': candidate_orders[order], 'records': records,
                                         'wall_ms': round((time.monotonic() - started_order) * 1000, 1)}
            row['grades'][order] = dict(case['screening']['grades'])
            row['usefulness'][order] = {identity: r['response']['answers']['probe_usefulness']['choice'] for identity, r in records.items()}
            print(case['scenario'], case['geometry'], order, row['grades'][order], row['usefulness'][order], flush=True)
        row['selection'] = selection(case['strategies'], row['usefulness'])
        row['wall_ms'] = round((time.monotonic() - started_case) * 1000, 1)
        rows.append(row)
        (folder / ('case-' + str(len(rows)) + '.json')).write_text(json.dumps(row, indent=2))
        print('SELECT', case['scenario'], case['geometry'], row['selection'], row['wall_ms'], flush=True)
    summary = {'kind': 'offline_local_screening_model_usefulness_three_permutations',
               'snapshots': len(cases) // 2, 'cases': len(rows),
               'requests': sum(len(e['records']) for r in rows for e in r['evaluations'].values()),
               'all_prompts_fit': True, 'all_tokenizer_ids_match': True, 'runtime_usage_matches': True,
               'permutations': list(grade_orders), 'groups': {}}
    for scenario in sorted({r['scenario'] for r in rows}):
        group = [r for r in rows if r['scenario'] == scenario]
        records = [record for r in group for e in r['evaluations'].values() for record in e['records'].values()]
        originals = [r for r in group if r['geometry'] == 'original']
        mirrors = {r['source_id']: r for r in group if r['geometry'] == 'mirror'}
        summary['groups'][scenario] = {
            'cases': len(group), 'candidate_order_assessments': len(records),
            'locally_eligible_candidate_geometry_pairs': sum(len(r['screening']['eligible']) for r in group),
            'locally_excluded_candidate_geometry_pairs': sum(len(r['screening']['excluded']) for r in group),
            'local_grade_source': 'deterministic_stage_fractions_not_model',
            'stable_usefulness_candidates': sum(len({g[identity] for g in r['usefulness'].values()}) == 1 for r in group for identity in next(iter(r['usefulness'].values()))),
            'candidate_geometry_pairs': sum(len(next(iter(r['grades'].values()))) for r in group),
            'choices': [{key: r[key] for key in ('source_id', 'geometry', 'selection', 'wall_ms')} for r in group],
            'mirror_selection_consistent_including_abstention': sum(mirrors[r['source_id']]['selection']['choice'] == swap_direction(r['selection']['choice']) for r in originals),
            'single_order_median_wall_ms': statistics.median(e['wall_ms'] for r in group for e in r['evaluations'].values()),
            'consensus_median_wall_ms': statistics.median(r['wall_ms'] for r in group),
            'local_preparation_median_ms': statistics.median(r['local_preparation_ms'] for r in group),
            'consensus_over_8s': sum(r['wall_ms'] > 8000 for r in group),
            'request_median_latency_ms': statistics.median(rec['latency_ms'] for rec in records),
            'max_prompt_tokens': max(q['tokens'] for rec in records for q in rec['prompt_audit'].values())}
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
