#!/usr/bin/env python3
"""Run fixed web-sim scenarios repeatedly and save the full baseline records."""
import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8766')
    parser.add_argument('--cases', nargs='+', default=['empty', 'open_target', 'l_turn', 'dead_end'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--output', default='tinynav_temp/navigation_lab/batch')
    args = parser.parse_args()
    if args.repeats < 1 or not 5 <= args.timeout <= 600:
        parser.error('repeats >= 1 and 5 <= timeout <= 600 are required')

    def api(path, body=None):
        request = Request(args.url.rstrip('/')+path, data=json.dumps(body).encode() if body is not None else None,
                          headers={'Content-Type':'application/json'})
        with urlopen(request, timeout=15) as response:
            return json.load(response)

    scenarios = api('/api/baseline/scenarios')['scenarios']
    if any(key not in scenarios for key in args.cases):
        parser.error('Unknown scenario; available: '+', '.join(scenarios))
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    results = []
    for key in args.cases:
        for repeat in range(args.repeats):
            scene = scenarios[key]; config = api('/api/default-config')
            config.update(name=scene['label'], scenario_id=key, start=scene['start'],
                          target=scene['target'], objects=scene['objects'])
            config['camera']['max_range'] = scene['cameraMaxRange']
            api('/api/baseline/start', {'config':config, 'timeout_s':args.timeout})
            deadline = time.monotonic()+args.timeout+30
            while True:
                metrics = api('/api/baseline/status')
                if metrics['status'] != 'running':
                    break
                if time.monotonic() > deadline:
                    api('/api/baseline/stop', {})
                    raise RuntimeError('Simulator did not finish by deadline')
                time.sleep(1)
            record = api('/api/baseline/export')
            (output/f"{key}-{repeat+1}-{metrics['id']}.json").write_text(json.dumps(record, indent=2))
            results.append({'scenario':key, **metrics})
            print(key, repeat+1, metrics['status'], f"{metrics['elapsed_s']:.1f}s", f"goal={metrics['distance_to_goal_m']:.2f}m", flush=True)
    (output/'summary.json').write_text(json.dumps(results, indent=2))
    print('Saved:', output.resolve(), flush=True)


if __name__ == '__main__':
    main()
