import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tinynav', 'core'))
from math_utils import uf_init, uf_find, uf_union, uf_all_sets_list

def generate_observations(N_frames, M_features):
    observations = []
    for f in range(N_frames):
        frame_obs = []
        for m in range(M_features):
            obs = (f, m, (float(f), float(m)))
            frame_obs.append(obs)
        observations.append(frame_obs)
    return observations


# benchmark disjoint set operations
def benchmark_disjoint_set(N_frames, M_features):
    observations = generate_observations(N_frames, M_features)

    N = N_frames * M_features
    parent, rank = uf_init(N)

    all_start_time = time.time()

    start_time = time.time()
    for frame_obs in observations[1:]:
        for obs in frame_obs:
            f, m, _ = obs
            idx = f * M_features + m
            idx_prev = (f - 1) * M_features + m
            uf_union(idx, idx_prev, parent, rank)
    end_time = time.time()
    print(f"Disjoint set union operations took {end_time - start_time:.4f} seconds for {len(observations)} observations.")

    start_time = time.time()
    for i in range(N):
        uf_find(i, parent)
    end_time = time.time()
    print(f"Disjoint set find operations took {end_time - start_time:.4f} seconds for {N} elements.")


    start_time = time.time()
    sets = uf_all_sets_list(parent)
    end_time = time.time()
    print(f"Collecting all sets took {end_time - start_time:.4f} seconds. Number of sets: {len(sets)}")

    all_end_time = time.time()
    print(f"Total disjoint set benchmark time: {all_end_time - all_start_time:.4f} seconds\n\n")


def bench(N_frames, M_features):
    for  _ in range(3):
        benchmark_disjoint_set(N_frames, M_features)

if __name__ == "__main__":
    bench(5, 10)
    bench(10, 200)
    bench(20, 200)
    bench(50, 1000)

