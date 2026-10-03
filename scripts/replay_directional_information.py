"""Audit exact prompts and replay presentation/mirror controls without execution."""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from tool.simulator.decision_observer import planning_questions
from tool.simulator.directional_information import explicit, mirror, compact_state_text, swap_direction
from scripts.replay_hierarchical import ask


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshots', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--runtime-source', required=True)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--repeats', type=int, default=2)
    a = p.parse_args()
    if a.repeats < 1:
        raise ValueError('repeats must be positive')
    sys.path.insert(0, a.runtime_source)
    from transformers import AutoTokenizer
    from startlux_decision import jevfmt as J
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    J.check_tokenizer(tok)
    with urlopen('http://127.0.0.1:8081/props') as response:
        props = json.load(response)
    capacity = props['default_generation_settings']['n_ctx']
    endpoint = 'http://127.0.0.1:8090/v1/systemone'
    folder = Path(a.output)
    folder.mkdir(parents=True, exist_ok=True)
    prompts = folder / 'prompts'
    prompts.mkdir(exist_ok=True)
    cases = []
    for path in sorted(Path(a.snapshots).glob('snapshot-*.json')):
        source = json.loads(path.read_text())
        original = source['evaluations']['v2']['request']['state']
        reflected = mirror(original)
        if mirror(reflected) != original:
            raise ValueError('Mirror is not an involution')
        case = {'source_id': source['source_id'], 'source_file': str(path), 'scenario': source['scenario'], 'variants': {}}
        for geometry, state in (('original', original), ('mirror', reflected)):
            questions = planning_questions(state)
            named = explicit(state)
            if planning_questions(named) != questions:
                raise ValueError('Questions changed between presentations')
            if [plan['stages'] for plan in named['recovery']['strategies']] != [plan['stages'] for plan in state['recovery']['strategies']]:
                raise ValueError('Commands changed')
            if json.loads(compact_state_text(state)) != J.annotate_indices(state):
                raise ValueError('Serialization changed state')
            variants = {'legacy': state, 'compact_csv': compact_state_text(state), 'explicit': compact_state_text(named)}
            for kind, evidence in variants.items():
                name = geometry + '-' + kind
                audit = {}
                for key, spec in questions.items():
                    row = J.from_systemone(evidence, spec, key)
                    messages, order = J.messages(row)
                    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    ids, _ = J.render_ids(row, tok, max_length=capacity - 1)
                    with urlopen(Request('http://127.0.0.1:8081/tokenize', data=json.dumps(
                            {'content': text, 'add_special': False}).encode(), headers={'Content-Type': 'application/json'})) as response:
                        server_ids = json.load(response)['tokens']
                    if ids != server_ids:
                        raise ValueError('Tokenizer mismatch')
                    file = path.stem + '-' + name + '-' + key + '.txt'
                    (prompts / file).write_text(text)
                    audit[key] = {'tokens': len(ids), 'tokenizer_parity': True, 'fits': True,
                                  'sha256': hashlib.sha256(text.encode()).hexdigest(), 'file': file, 'option_order': order}
                case['variants'][name] = {'request': {'state': evidence, 'questions': questions}, 'prompt_audit': audit}
        cases.append(case)
    if not cases:
        raise ValueError('No snapshots')
    (folder / 'preflight.json').write_text(json.dumps({'capacity': capacity, 'cases': cases}, indent=2))
    print('PREFLIGHT', len(cases), 'snapshots, all prompts fit and tokenizer IDs match', flush=True)
    rows = []
    names = ['original-legacy', 'original-compact_csv', 'original-explicit',
             'mirror-legacy', 'mirror-compact_csv', 'mirror-explicit']
    for repeat in range(a.repeats):
        for index, case in enumerate(cases):
            shift = (index + repeat * 3) % len(names)
            order = names[shift:] + names[:shift]
            row = {key: case[key] for key in ('source_id', 'source_file', 'scenario')}
            row.update(repeat=repeat + 1, order=order, evaluations={})
            for name in order:
                variant = case['variants'][name]
                result = ask(variant['request']['state'], variant['request']['questions'], endpoint)
                expected = sum(item['tokens'] for item in variant['prompt_audit'].values())
                if result['response']['usage']['input_tokens'] != expected:
                    raise ValueError('Runtime rendering differs from audited rendering')
                result['prompt_audit'] = variant['prompt_audit']
                result['choice'] = result['response']['answers']['alternative']['choice']
                row['evaluations'][name] = result
                print(row['scenario'], row['repeat'], name, result['choice'], result['latency_ms'], flush=True)
            rows.append(row)
            (folder / ('pair-' + str(len(rows)) + '.json')).write_text(json.dumps(row, indent=2))
    summary = {'kind': 'offline_same_snapshot_presentation_and_reflection_no_execution',
               'snapshots': len(cases), 'repeats': a.repeats, 'rows': len(rows), 'requests': len(rows) * len(names),
               'capacity': capacity, 'all_prompts_fit': True, 'all_tokenizer_ids_match': True,
               'all_runtime_usage_matches_prompt_audit': True, 'groups': {}}
    for scenario in sorted({r['scenario'] for r in rows}):
        group = [r for r in rows if r['scenario'] == scenario]
        data = {}
        for kind in ('legacy', 'compact_csv', 'explicit'):
            pairs = [(r['evaluations']['original-' + kind], r['evaluations']['mirror-' + kind]) for r in group]
            flipped = sum(b['choice'] == swap_direction(a['choice']) for a, b in pairs)
            non_abstain = [(a, b) for a, b in pairs if a['choice'].startswith('strategy_')]
            # Abstaining in both scenes is symmetry consistency, not evidence of direction discrimination.
            deltas = []
            for first, second in pairs:
                a_probs = first['response']['answers']['alternative']['probabilities']
                b_probs = second['response']['answers']['alternative']['probabilities']
                mapped = {swap_direction(k): v for k, v in a_probs.items()}
                if set(mapped) != set(b_probs):
                    raise ValueError('Mirror choices do not correspond')
                deltas.append(sum(abs(mapped[k] - b_probs[k]) for k in mapped))
            evaluations = [item for pair in pairs for item in pair]
            data[kind] = {'original_choices': [a['choice'] for a, b in pairs],
                          'mirror_choices': [b['choice'] for a, b in pairs],
                          'mirror_consistent_including_abstention': flipped,
                          'directional_original_count': len(non_abstain),
                          'directional_mirror_consistent': sum(b['choice'] == swap_direction(a['choice']) for a, b in non_abstain),
                          'mean_mirrored_distribution_l1': statistics.mean(deltas),
                          'median_latency_ms': statistics.median(e['latency_ms'] for e in evaluations),
                          'max_latency_ms': max(e['latency_ms'] for e in evaluations),
                          'over_8s': sum(e['latency_ms'] > 8000 for e in evaluations),
                          'max_prompt_tokens': max(q['tokens'] for e in evaluations for q in e['prompt_audit'].values())}
        data['original_changed_by_whitespace'] = sum(r['evaluations']['original-legacy']['choice'] != r['evaluations']['original-compact_csv']['choice'] for r in group)
        data['original_changed_by_named_fields'] = sum(r['evaluations']['original-compact_csv']['choice'] != r['evaluations']['original-explicit']['choice'] for r in group)
        summary['groups'][scenario] = data
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
