import sys
import time
from collections import OrderedDict

def _nbytes(obj):
    nbytes = getattr(obj, 'nbytes', None)
    if nbytes is not None:
        return int(nbytes)
    if isinstance(obj, dict):
        return int(sys.getsizeof(obj)) + sum((_nbytes(k) + _nbytes(v) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return int(sys.getsizeof(obj)) + sum((_nbytes(v) for v in obj))
    return int(sys.getsizeof(obj))

class _LRUBase:

    def __deepcopy__(self, memo):
        cls = type(self)
        fresh = cls.__new__(cls)
        memo[id(self)] = fresh
        fresh._cap = self._cap
        fresh._d = type(self._d)()
        fresh._bytes = 0
        fresh.hits = 0
        fresh.misses = 0
        fresh.evicted = 0
        return fresh
    __slots__ = ('_cap', '_d', '_bytes', 'hits', 'misses', 'evicted')

    def __init__(self, cap_bytes):
        self._cap = max(0, int(cap_bytes))
        self._d = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evicted = 0

    @property
    def enabled(self):
        return self._cap > 0

    @property
    def nbytes(self):
        return self._bytes

    def stats(self):
        return {'entries': len(self._d), 'bytes': self._bytes, 'hits': self.hits, 'misses': self.misses, 'evicted': self.evicted, 'cap_bytes': self._cap}

    def get(self, key, default=None):
        if not self.enabled:
            return default
        try:
            val = self._d.pop(key)
        except KeyError:
            self.misses += 1
            return default
        self._d[key] = val
        self.hits += 1
        return val

    def put(self, key, value):
        if not self.enabled:
            return
        size = self._size(value)
        if size > self._cap:
            return
        old = self._d.pop(key, None)
        if old is not None:
            self._bytes -= self._size(old)
        self._d[key] = value
        self._bytes += size
        while self._bytes > self._cap and self._d:
            _k, _v = self._d.popitem(last=False)
            self._bytes -= self._size(_v)
            self.evicted += 1

    def clear(self):
        self._d.clear()
        self._bytes = 0

    def _size(self, value):
        return _nbytes(value)

class ResultCache(_LRUBase):
    __slots__ = ()

    @staticmethod
    def key(x, tag='', ndigits=9):
        try:
            return (tag,) + tuple((round(float(v), ndigits) for v in x))
        except TypeError:
            return None

    def _size(self, value):
        F, G, info = value
        return _nbytes(F) + _nbytes(G) + _nbytes(info) + 128

class StateCache(_LRUBase):
    __slots__ = ()
    _FIELDS = ('T', 'rho', 'h', 'G', 'p', 'T_wall', 'alpha', 'htc_film')

    def _size(self, value):
        nbytes = getattr(value, 'nbytes', None)
        if nbytes is not None:
            return int(nbytes)
        total = 0
        for name in self._FIELDS:
            arr = getattr(value, name, None)
            if arr is not None:
                total += _nbytes(arr)
        return total or _nbytes(value)

class CachedCase:

    def __init__(self, cap_bytes):
        self.cache = ResultCache(cap_bytes)
        self._t0 = time.time()

    def lookup(self, x, tag=''):
        key = ResultCache.key(x, tag)
        if key is None:
            return None
        return self.cache.get(key)

    def store(self, x, result, tag=''):
        key = ResultCache.key(x, tag)
        if key is not None:
            self.cache.put(key, result)

    def preload(self, items):
        n = 0
        for key, val in items:
            if key is None:
                continue
            self.cache.put(key, val)
            n += 1
        return n

    def summary(self):
        s = self.cache.stats()
        total = s['hits'] + s['misses']
        s['hit_rate'] = s['hits'] / total if total else 0.0
        s['elapsed_s'] = time.time() - self._t0
        return s

def load_presolved(src, tag='', n_obj=None, n_constr=None, objective_tag=None):
    import json
    import os
    records = []
    if isinstance(src, (list, tuple)):
        records = list(src)
    elif str(src).lower().endswith('.npz'):
        import numpy as np
        z = np.load(src, allow_pickle=True)
        X, F = (z['X'], z['F'])
        G = z['G'] if 'G' in z else None
        for i in range(len(X)):
            records.append({'x': X[i], 'F': F[i], 'G': G[i] if G is not None else []})
    else:
        with open(src, encoding='utf-8') as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            meta = data.get('meta') or {}
            if objective_tag and meta.get('objective_tag') not in (None, objective_tag):
                return []
            records = data.get('lib', data)
        else:
            records = data
    out = []
    for rec in records:
        x = rec.get('x')
        if x is None:
            continue
        F = rec.get('F')
        if F is None and 'S' in rec and (n_obj in (None, 2)):
            F = [rec['S'], rec.get('V', 0.0)]
        if F is None:
            continue
        if n_obj is not None and len(F) != int(n_obj):
            continue
        G = rec.get('G', rec.get('g', []))
        if not len(G):
            G = [-1.0] * int(n_constr or 5)
        if n_constr is not None and len(G) != int(n_constr):
            continue
        info = rec.get('info')
        if not isinstance(info, dict):
            info = {'prerun': True}
        key = ResultCache.key(x, tag)
        if key is None:
            continue
        out.append((key, (list(map(float, F)), list(map(float, G)), info)))
    return out
