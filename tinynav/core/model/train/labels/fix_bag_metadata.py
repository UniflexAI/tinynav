"""Make a readable copy of a sqlite3 ros2 bag whose metadata.yaml is missing or broken.

The copy is a new folder with a symlink to the original .db3 and a metadata.yaml rebuilt from the database,
so the original recording is never modified.

  python -m tinynav.core.model.train.labels.fix_bag_metadata <bag_dir> <out_dir>
"""
import argparse
import sqlite3
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('out')
    args = ap.parse_args()
    src, out = Path(args.bag).resolve(), Path(args.out)
    dbs = sorted(src.glob('*.db3'))
    out.mkdir(parents=True, exist_ok=True)
    topics, t0, t1, total = {}, None, None, 0
    for db in dbs:
        (out / db.name).unlink(missing_ok=True)
        (out / db.name).symlink_to(db)
        con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
        for tid, name, typ, fmt, qos in con.execute('select id, name, type, serialization_format, offered_qos_profiles from topics'):
            n, lo, hi = con.execute('select count(*), min(timestamp), max(timestamp) from messages where topic_id=?', (tid,)).fetchone()
            topics.setdefault(name, dict(type=typ, fmt=fmt, qos=qos, n=0))['n'] += n
            total += n
            if n:
                t0 = lo if t0 is None else min(t0, lo)
                t1 = hi if t1 is None else max(t1, hi)
        con.close()
    meta = dict(rosbag2_bagfile_information=dict(
        version=5, storage_identifier='sqlite3', relative_file_paths=[db.name for db in dbs],
        duration=dict(nanoseconds=int(t1 - t0)), starting_time=dict(nanoseconds_since_epoch=int(t0)),
        message_count=total, compression_format='', compression_mode='',
        topics_with_message_count=[dict(topic_metadata=dict(name=k, type=v['type'], serialization_format=v['fmt'],
                                                            offered_qos_profiles=v['qos'] or ''), message_count=v['n'])
                                   for k, v in topics.items()]))
    (out / 'metadata.yaml').write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f'{out}: {len(topics)} topics, {total} messages, {(t1 - t0) / 1e9:.0f} s')


if __name__ == '__main__':
    main()
