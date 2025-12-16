from async_lru import alru_cache
from functools import wraps
import numpy as np
import hashlib


LARGE_THRESHOLD = 10_000   # if array.size > this → partial hash
STEP = 10                  # hash 1/10 of elements

def _hash_ndarray(a: np.ndarray):
    """
    Fast hashing strategy:
      - full hash if array has <= 10k elements
      - partial hash (1/10 sampling) if larger
    Ensures the buffer we give to memoryview is C-contiguous.
    """
    a_c = np.ascontiguousarray(a)

    if a_c.size > LARGE_THRESHOLD:
        # Partial: 1/10 of elements, then force that small slice contiguous
        sampled = np.ascontiguousarray(a_c.ravel()[::STEP])
        view = memoryview(sampled)
        mode = STEP    # include mode so keys differ if strategy changes
    else:
        # Small arrays: hash entire contiguous buffer
        view = memoryview(a_c)
        mode = 1       # full hash

    digest = hashlib.sha1(view).digest()
    return (a_c.shape, str(a_c.dtype), mode, digest)


def _make_hash_key(args, kwargs):
    """
    Build a hashable key assuming:
      - args may contain numpy arrays but not nested lists/tuples/dicts
      - kwargs same assumption
    """
    key_args = tuple(
        _hash_ndarray(x) if isinstance(x, np.ndarray) else x
        for x in args
    )

    key_kwargs = tuple(
        sorted(
            (k, _hash_ndarray(v) if isinstance(v, np.ndarray) else v)
            for k, v in kwargs.items()
        )
    )

    return (key_args, key_kwargs)


def alru_cache_numpy(maxsize=128, **alru_kwargs):
    """
    Async LRU cache that supports numpy arrays,
    using the hashing rules above.
    """
    def decorator(func):
        # Stores hash_key → (real args, real kwargs)
        cache_store = {}

        @alru_cache(maxsize=maxsize, **alru_kwargs)
        async def _cached(hash_key):
            real_args, real_kwargs = cache_store[hash_key]
            return await func(*real_args, **real_kwargs)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            hash_key = _make_hash_key(args, kwargs)
            remove_keys = [
                k
                for k in cache_store.keys() if k not in _cached._LRUCacheWrapper__cache.keys()
            ]
            for k in remove_keys:
                del cache_store[k]

            if hash_key not in cache_store:
                cache_store[hash_key] = (args, kwargs)

            return await _cached(hash_key)

        wrapper.cache_info = _cached.cache_info
        wrapper.cache_clear = _cached.cache_clear
        return wrapper

    return decorator
