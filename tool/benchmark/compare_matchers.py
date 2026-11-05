"""Read-only keyframe-pair replay and GIF export for LightGlue/FlexGlue."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import shelve
import statistics
import time

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from tinynav.core.flexglue import FlexGlueONNX
from tinynav.core.math_utils import rerank_by_pnp_inliers
from tinynav.core.models_trt import LightGlueTRT
from tool.video_db import VideoDB


def load_frame(root, identity):
    name, key = identity.split('/')
    directory = root / name
    with shelve.open(str(directory / 'features'), flag='r') as store: features = store[key]
    with shelve.open(str(directory / 'depths'), flag='r') as store: depth = store[key]
    video = VideoDB(str(directory / 'infra1_images_db'), mode='read')
    try: image = video.read(int(key))
    finally: video.close()
    if image is None: raise ValueError(f'Image missing for {identity}')
    if image.ndim == 3: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    poses = np.load(directory / 'poses.npy', allow_pickle=True).item()
    return dict(name=name, key=key, image=image, features=features, depth=depth,
                K=np.load(directory / 'intrinsics.npy'), pose=poses[int(key)])


def solve(reference, query, matches, transforms):
    r, q = reference, query
    p0 = r['features']['kpts'][0][matches[:, 0]]
    p1 = q['features']['kpts'][0][matches[:, 1]]
    u, v = p0.astype(np.int64).T
    valid = (u >= 0) & (v >= 0) & (u < r['depth'].shape[1]) & (v < r['depth'].shape[0])
    u, v, p1 = u[valid], v[valid], p1[valid]
    z = r['depth'][v, u]; good = np.isfinite(z) & (z > 0) & (z < 50)
    u, v, z, p1 = u[good], v[good], z[good], p1[good]
    K = r['K']; xyz = np.c_[(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z]
    pose = np.asarray(transforms.get(r['name'], np.eye(4))) @ r['pose']
    xyz = xyz @ pose[:3, :3].T + pose[:3, 3]
    cv2.setRNGSeed(17)
    ok, estimated, _, _, inliers, _ = rerank_by_pnp_inliers([(xyz, p1)], q['K'])
    result = {'accepted': bool(ok), 'inliers': int(inliers), 'valid_depth_matches': len(p1)}
    comparable = r['name'] == q['name'] or all(n in transforms for n in (r['name'], q['name']))
    if ok and comparable:
        truth = np.asarray(transforms.get(q['name'], np.eye(4))) @ q['pose']
        measured = np.linalg.inv(estimated)
        result['position_error_m'] = float(np.linalg.norm(measured[:3, 3] - truth[:3, 3]))
        result['rotation_error_deg'] = float(np.rad2deg(Rotation.from_matrix(truth[:3, :3].T @ measured[:3, :3]).magnitude()))
    return result


def export_media(reference, query, results, output):
    images = [reference['image'], query['image']]
    frames = []
    for count in range(17):
        canvas = np.full((540, 1120, 3), 22, np.uint8)
        cv2.putText(canvas, 'TinyNav | repetitive-corridor matcher comparison', (18, 28), 0, .75, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, 'Offline keyframe pair | same SuperPoint, depth and PnP acceptance rules', (18, 54), 0, .51, (190, 190, 190), 1, cv2.LINE_AA)
        for panel, (label, item) in enumerate(results.items()):
            left = 15 + panel * 560
            cv2.putText(canvas, label, (left, 88), 0, .66, (245, 245, 245), 1, cv2.LINE_AA)
            origins = []; scales = []
            for column, image in enumerate(images):
                scale = min(250 / image.shape[1], 290 / image.shape[0]); w, h = int(image.shape[1] * scale), int(image.shape[0] * scale)
                x, y = left + column * 270, 117
                canvas[y:y+h, x:x+w] = cv2.cvtColor(cv2.resize(image, (w, h)), cv2.COLOR_GRAY2BGR)
                cv2.putText(canvas, 'Reference' if column == 0 else 'Query', (x, 108), 0, .4, (180, 180, 180), 1, cv2.LINE_AA)
                origins.append(np.array([x, y]));scales.append(scale)
            matches = item['matches']; selected = matches[::max(1, len(matches)//50)][:50]
            for k, (a, b) in enumerate(selected[:int(len(selected) * count / 16)]):
                p0 = reference['features']['kpts'][0, a] * scales[0] + origins[0]
                p1 = query['features']['kpts'][0, b] * scales[1] + origins[1]
                cv2.line(canvas, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), (70, 205, 150), 1, cv2.LINE_AA)
            lines = [f"Raw matches: {len(matches)} | match: {item['matching_ms']:.1f} ms",
                     f"PnP: {'ACCEPTED' if item['accepted'] else 'REJECTED'} | inliers: {item['inliers']}"]
            if 'position_error_m' in item:
                lines.append(f"Map-ref error: {item['position_error_m']:.2f} m / {item['rotation_error_deg']:.1f} deg")
            else: lines.append('Map-ref error: not available' if item['accepted'] else 'No accepted pose')
            for i, line in enumerate(lines):cv2.putText(canvas, line, (left, 438 + i*26), 0, .5, (225, 225, 225), 1, cv2.LINE_AA)
        cv2.putText(canvas, 'Lines reveal raw matches, not PnP inliers or robot motion. Map reference is not surveyed ground truth.', (18, 529), 0, .44, (165, 165, 165), 1, cv2.LINE_AA)
        frames.append(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).quantize(colors=96))
    frames[-1].convert('RGB').save(output / 'comparison.png')
    frames[0].save(output / 'comparison.gif', save_all=True, append_images=frames[1:], duration=[90]*16+[1500], loop=0, optimize=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--maps_root', type=Path, required=True)
    p.add_argument('--reference', required=True, help='Map/keyframe_timestamp')
    p.add_argument('--query', required=True, help='Map/keyframe_timestamp')
    p.add_argument('--transforms', type=Path, help='JSON map-name to common-frame transform, or optimization.json')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args();args.output.mkdir(parents=True, exist_ok=True)
    transforms = json.loads(args.transforms.read_text()) if args.transforms else {}
    transforms = transforms.get('optimized_transforms', transforms)
    r, q = load_frame(args.maps_root, args.reference), load_frame(args.maps_root, args.query)
    light, flex = LightGlueTRT(), FlexGlueONNX()
    qa, ra = q['features'], r['features']
    fqa, fra = flex.add_features(qa, q['image']), flex.add_features(ra, r['image'])
    loop = asyncio.new_event_loop()
    def run_light():
        light.infer.cache_clear()
        output = loop.run_until_complete(light.infer(ra['kpts'], qa['kpts'], ra['descps'], qa['descps'], ra['mask'], qa['mask'], np.array(r['image'].shape[::-1], np.int64), np.array(q['image'].shape[::-1], np.int64)))
        indices = output['match_indices'][0];valid = np.flatnonzero(indices >= 0)
        return np.column_stack((valid, indices[valid])).astype(np.int64)
    results = {}
    for name, run in [('LightGlue / TensorRT', run_light), ('FlexGlue / ONNX CUDA', lambda:flex.match_keypoints(fra, fqa)[2])]:
        for _ in range(3): run()
        times=[]
        for _ in range(7):
            start=time.perf_counter();matches=run();times.append((time.perf_counter()-start)*1000)
        results[name] = dict(matches=matches, matching_ms=statistics.median(times), **solve(r, q, matches, transforms))
    export_media(r, q, results, args.output)
    output = {name:{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in item.items()} for name,item in results.items()}
    output['input'] = {'reference': args.reference, 'query': args.query, 'image_sha256': [hashlib.sha256(x['image'].tobytes()).hexdigest() for x in (r,q)], 'note': 'Fixed pair; cached descriptors; match-only timings; map-derived reference, not surveyed truth.'}
    (args.output/'comparison.json').write_text(json.dumps(output, indent=2));loop.close()


if __name__ == '__main__':main()
