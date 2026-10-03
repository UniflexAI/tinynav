"""Save exact local adapter rendering and verify token parity against llama.cpp."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen


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
        props = json.load(response)
    capacity = props['default_generation_settings']['n_ctx']
    folder = Path(a.output)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(Path(a.snapshots).glob('snapshot-*.json')):
        source = json.loads(path.read_text())
        payload = source['evaluations']['v2']['request']
        for key, spec in payload['questions'].items():
            row = J.from_systemone(payload['state'], spec, key)
            msgs, order = J.messages(row)
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids, actual_order = J.render_ids(row, tok, max_length=65536)
            if actual_order != order:
                raise ValueError('Option order changed')
            with urlopen(Request('http://127.0.0.1:8081/tokenize', data=json.dumps(
                    {'content': text, 'add_special': False}).encode(), headers={'Content-Type': 'application/json'})) as response:
                server_ids = json.load(response)['tokens']
            if ids != server_ids:
                raise ValueError('Tokenizer mismatch')
            name = path.stem + '-' + key
            (folder / (name + '.txt')).write_text(text)
            (folder / (name + '-ids.json')).write_text(json.dumps(ids))
            rows.append({'source_id': source['source_id'], 'scenario': source['scenario'],
                         'question': key, 'tokens': len(ids), 'slot_capacity': capacity,
                         'fits_with_one_output_token': len(ids) + 1 <= capacity,
                         'tokenizer_parity': True, 'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(),
                         'option_order': order, 'state_exact': row['state'] == J.state_text(payload['state']),
                         'thinking_off_prefix': text.endswith(J.THINK_OFF_SUFFIX), 'file': name + '.txt'})
    report = {'slot_capacity': capacity, 'slots': props['total_slots'], 'prompts': rows,
              'range_tokens': [min(r['tokens'] for r in rows), max(r['tokens'] for r in rows)],
              'all_fit': all(r['fits_with_one_output_token'] for r in rows),
              'usage_is_sum_of_independent_question_prompts': True,
              'runtime_sources_sha256': {name: hashlib.sha256((Path(a.runtime_source) / 'startlux_decision' / name).read_bytes()).hexdigest()
                                         for name in ('gguf_server.py', 'jevfmt.py', 'model.py')}}
    (folder / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'prompts'}, indent=2))


if __name__ == '__main__':
    main()
