import csv
import json
import multiprocessing as mp
import os
import time
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pyht.optim.cache import CachedCase

def _make_nsga3(spec, extra_kw):
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.util.ref_dirs import get_reference_directions
    M = max(2, int(spec.n_obj))
    pop = max(M + 1, int(spec.pop_size))
    n_dirs, part = _ref_dirs_for(M, pop)
    if n_dirs * 1 != pop:
        new_pop = n_dirs * max(1, pop // n_dirs)
        print(f'  [{spec.name}] NSGA-III: pop_size {pop} -> {new_pop} (must be a multiple of {n_dirs} ref dirs)', flush=True)
        spec.pop_size = new_pop
        pop = new_pop
    ref = get_reference_directions('das-dennis', M, n_partitions=part)
    print(f'  [{spec.name}] NSGA-III: pop={pop}, M={M}, ref_dirs={len(ref)} (das-dennis p={part})', flush=True)
    kw = dict(pop_size=pop, ref_dirs=ref)
    kw.update(extra_kw)
    return NSGA3(**kw)

def _ref_dirs_for(n_obj, pop_size):
    from pymoo.util.ref_dirs import get_reference_directions
    best = None
    for part in range(1, 25):
        n = len(get_reference_directions('das-dennis', n_obj, n_partitions=part))
        if n == 1:
            continue
        if pop_size % n == 0:
            best = (n, part)
    if best is not None:
        return best
    cand = []
    for part in range(1, 25):
        n = len(get_reference_directions('das-dennis', n_obj, n_partitions=part))
        if n <= pop_size:
            cand.append((n, part))
    return cand[-1] if cand else (1, 1)

class IndexRound(Repair):

    def __init__(self, integer_vars=()):
        super().__init__()
        self.integer_vars = tuple(integer_vars)

    def _do(self, problem, X, **kwargs):
        for i in self.integer_vars:
            X[:, i] = np.round(X[:, i])
        return X

class CaseProblem(Problem):

    def __init__(self, spec):
        super().__init__(n_var=len(spec.bounds), n_obj=spec.n_obj, n_ieq_constr=spec.n_constr, xl=np.array([b[0] for b in spec.bounds], dtype=float), xu=np.array([b[1] for b in spec.bounds], dtype=float))
        self._spec = spec
        self._res, _state_bytes = spec.cache_split()
        self._cached = CachedCase(self._res)
        from pyht.optim.cache import StateCache
        _per_worker = max(int(_state_bytes), 0) * max(1, int(spec.n_jobs))
        _two_gen = int(spec.pop_size) * 2 * _STATE_MB * 2 ** 20
        _cap = max(_per_worker, min(_two_gen, int(spec.memory_budget_bytes) // 2))
        self._states = StateCache(_cap)
        _SIG_FN[0] = None
        _COST_FN[0] = None
        try:
            import importlib
            _mod = importlib.import_module(spec.worker_module)
            _SIG_FN[0] = getattr(_mod, 'warm_signature', None)
            _COST_FN[0] = getattr(_mod, 'cost_hint', None)
        except Exception:
            pass
        self._tag = json.dumps({'module': spec.worker_module, 'kw': spec.solver_kwargs, 'obj': getattr(spec, 'objective_tag', ''), 'n_obj': spec.n_obj, 'n_constr': spec.n_constr}, sort_keys=True, default=str)
        self.n_presolved = 0
        if spec.presolved:
            from pyht.optim.cache import load_presolved
            try:
                items = load_presolved(spec.presolved, self._tag, n_obj=spec.n_obj, n_constr=spec.n_constr, objective_tag=getattr(spec, 'objective_tag', ''))
                self.n_presolved = self._cached.preload(items)
            except Exception as exc:
                print(f'  [{spec.name}] presolved 读取失败，忽略: {exc}', flush=True)

    def _evaluate(self, X, out, *args, **kwargs):
        from pyht.optim.worker import eval_batch, eval_generic
        rows = list(X)
        results = [None] * len(rows)
        todo = []
        for i, x in enumerate(rows):
            hit = self._cached.lookup(x, self._tag)
            if hit is not None:
                results[i] = hit
            else:
                todo.append(i)
        if todo:
            xs = [rows[i] for i in todo]
            if getattr(self._spec, 'warm_start', False):
                fresh, _states = self._eval_with_warm(xs)
            else:
                fresh = self._dispatch(eval_generic, xs)
            for i, x, r in zip(todo, xs, fresh):
                results[i] = r
                self._cached.store(x, r, self._tag)
        out['F'] = np.array([r[0] for r in results], dtype=float)
        if self._spec.n_constr > 0:
            out['G'] = np.array([r[1] for r in results], dtype=float)
        out['_info'] = [r[2] for r in results]

    def _costs(self, xs):
        fn = _COST_FN[0]
        if fn is None:
            return [0.0] * len(xs)
        out = []
        for x in xs:
            try:
                out.append(float(fn(x)))
            except Exception:
                out.append(0.0)
        return out

    def _map_lpt(self, fn, items, costs):
        n = len(items)
        if n <= 1:
            return _get_pool().map(fn, items)
        order = sorted(range(n), key=lambda i: -costs[i])
        got = _get_pool().map(fn, [items[i] for i in order], chunksize=1)
        out = [None] * n
        for pos, i in enumerate(order):
            out[i] = got[pos]
        return out

    def _dispatch(self, fn, xs):
        return self._map_lpt(fn, xs, self._costs(xs))

    def _sig(self, x):
        fn = _SIG_FN[0]
        if fn is None:
            return None
        try:
            return fn(x)
        except Exception:
            return None

    def _eval_with_warm(self, xs):
        from pyht.optim.worker import eval_warm
        sigs = [self._sig(x) for x in xs]
        hand = [self._states.get(k) if k is not None else None for k in sigs]
        out = [None] * len(xs)
        states = [None] * len(xs)
        costs = self._costs(xs)
        todo = list(range(len(xs)))
        got = self._map_lpt(eval_warm, [(xs[i], hand[i]) for i in todo], costs)
        cold_again = []
        for i, (res, st) in zip(todo, got):
            out[i] = res
            states[i] = st
            if st is None and sigs[i] is not None:
                cold_again.append(i)
        if cold_again:
            retry = []
            for i in cold_again:
                st = self._states.get(sigs[i])
                if st is not None:
                    retry.append((i, st))
            if retry:
                got2 = self._map_lpt(eval_warm, [(xs[i], st) for i, st in retry], [costs[i] for i, _ in retry])
                for (i, _), (res, st) in zip(retry, got2):
                    out[i] = res
                    if st is not None:
                        states[i] = st
        for i, st in enumerate(states):
            if st is not None and sigs[i] is not None:
                self._states.put(sigs[i], st)
        return (out, states)

    def cache_summary(self):
        return self._cached.summary()
_POOL = None
_N_WORKERS = [1]
_SIG_FN = [None]
_COST_FN = [None]
_STATE_MB = 12

def _warm_summary(problem=None):
    if problem is None:
        return {}
    c = problem._states
    n = c.hits + c.misses
    return {'hits': c.hits, 'misses': c.misses, 'entries': len(c._d), 'bytes': c.nbytes, 'enabled': c.enabled, 'evicted': c.evicted, 'cap_bytes': c._cap, 'hit_rate': c.hits / n if n else 0.0}

def _sig_sort_key(sig_fn, x):
    try:
        return repr(sig_fn(x))
    except Exception:
        return ''

def _get_pool():
    if _POOL is None:
        raise RuntimeError('worker pool not initialised (call run_nsga2)')
    return _POOL

def metrics_header(spec):
    header = ['gen', 'n_eval', 'n_feasible']
    for i in range(spec.n_obj):
        header += [f'obj{i}_min', f'obj{i}_max', f'obj{i}_mean']
    for v in spec.var_names:
        header += [f'{v}_min', f'{v}_max', f'{v}_mean', f'{v}_std']
    header.append('gen_t')
    header += ['cache_hit', 'cache_miss', 'cache_mb', 'warm_hit', 'warm_miss']
    return header

def _compute_metrics(spec, gen, pop, gen_t):
    X = pop.get('X')
    F = pop.get('F')
    G = pop.get('G')
    if X is None or F is None:
        return None
    n = len(F)
    if G is not None and G.size and (spec.n_constr > 0):
        n_feasible = int((G.max(axis=1) <= 0).sum())
    else:
        n_feasible = n
    metrics = {'gen': gen, 'n_eval': gen * spec.pop_size, 'n_feasible': n_feasible}
    for i in range(spec.n_obj):
        col = F[:, i]
        metrics[f'obj{i}_min'] = float(col.min())
        metrics[f'obj{i}_max'] = float(col.max())
        metrics[f'obj{i}_mean'] = float(col.mean())
    for i, v in enumerate(spec.var_names):
        col = X[:, i]
        metrics[f'{v}_min'] = float(col.min())
        metrics[f'{v}_max'] = float(col.max())
        metrics[f'{v}_mean'] = float(col.mean())
        metrics[f'{v}_std'] = float(col.std())
    metrics['gen_t'] = gen_t
    return metrics

def _collect_details(info_list):
    if info_list is None:
        return {}
    info_list = list(info_list)
    if not info_list or info_list[0] is None:
        return {}
    keys = set(info_list[0].keys())
    for info in info_list[1:]:
        if info is None:
            return {}
        keys &= set(info.keys())
    numeric = (int, float, bool, np.integer, np.floating, np.bool_)
    details = {}
    for k in sorted(keys):
        vals = [info.get(k) for info in info_list]
        if all((isinstance(v, numeric) for v in vals)):
            details[k] = np.asarray(vals)
    return details

def latest_checkpoint(checkpoint_dir, upto=None):
    best = None
    if not os.path.isdir(checkpoint_dir):
        return None
    for name in os.listdir(checkpoint_dir):
        if not (name.startswith('gen_') and name.endswith('.npz')):
            continue
        try:
            gen = int(name[4:-4])
        except ValueError:
            continue
        if upto is not None and gen >= upto:
            continue
        if best is None or gen > best[0]:
            best = (gen, os.path.join(checkpoint_dir, name))
    return best

def load_checkpoint(path):
    d = np.load(path, allow_pickle=True)
    X = np.asarray(d['X'], dtype=float)
    F = np.asarray(d['F'], dtype=float)
    G = np.asarray(d['G'], dtype=float) if 'G' in d else None
    info = None
    if 'details' in d:
        det = d['details'].item()
        if det:
            keys = list(det)
            info = [{k: det[k][i] for k in keys} for i in range(len(X))]
    return (X, F, G, info)

def save_checkpoint(spec, gen, X, F, G, info_list):
    os.makedirs(spec.checkpoint_dir, exist_ok=True)
    payload = {'X': np.asarray(X), 'F': np.asarray(F)}
    if G is not None:
        payload['G'] = np.asarray(G)
    details = _collect_details(info_list)
    if details:
        payload['details'] = np.array(details, dtype=object)
    np.savez(os.path.join(spec.checkpoint_dir, f'gen_{gen:04d}.npz'), **payload)
    return details

def append_metrics_csv(spec, metrics, header):
    fp = os.path.join(spec.checkpoint_dir, 'metrics.csv')
    write_header = not os.path.isfile(fp)
    with open(fp, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow({k: metrics.get(k, '') for k in header})

def run_nsga2(spec, *, sampling=None, callback=None, resume=False, save_history=False, wall_budget_s=None):
    from pyht.optim.worker import init_worker
    if spec.table_ranges:
        from pyht.fluid import CoolPropWrapper
        for kwargs in spec.table_ranges:
            CoolPropWrapper.set_table_range(**kwargs)
    os.makedirs(spec.checkpoint_dir, exist_ok=True)
    header = metrics_header(spec)
    _meta_path = os.path.join(spec.checkpoint_dir, 'run_meta.json')
    _meta = {'objective_tag': getattr(spec, 'objective_tag', ''), 'n_obj': int(spec.n_obj), 'n_constr': int(spec.n_constr), 'worker_module': spec.worker_module, 'solver_kwargs': dict(spec.solver_kwargs or {})}
    if os.path.isfile(_meta_path):
        with open(_meta_path, encoding='utf-8') as fh:
            _old = json.load(fh)
        _keys = ('objective_tag', 'n_obj', 'n_constr', 'worker_module')
        _bad = {k: (_old.get(k), _meta[k]) for k in _keys if _old.get(k) != _meta[k]}
        if _bad:
            raise RuntimeError('checkpoint 目录 %s 的 run_meta 与当前运行不一致：%r。\n旧 checkpoint 的目标/约束语义不同，不能续跑或覆盖；请换一个 --out。' % (spec.checkpoint_dir, _bad))
    else:
        import time as _time
        _meta['created'] = _time.strftime('%Y-%m-%dT%H:%M:%S')
        with open(_meta_path, 'w', encoding='utf-8') as fh:
            json.dump(_meta, fh, ensure_ascii=False, indent=1, default=str)
    global _POOL
    _N_WORKERS[0] = max(1, int(spec.n_jobs))
    _res_bytes, _state_bytes = spec.cache_split()
    _POOL = mp.Pool(spec.n_jobs, initializer=init_worker, initargs=(spec.worker_module, _state_bytes, dict(spec.solver_kwargs or {})))
    if _res_bytes or _state_bytes:
        print(f'  [{spec.name}] cache: result {_res_bytes / 2 ** 20:.0f} MB, state {_state_bytes / 2 ** 20:.0f} MB/worker x {spec.n_jobs} = {_state_bytes * spec.n_jobs / 2 ** 20:.0f} MB (budget {spec.memory_budget_bytes / 2 ** 30:.1f} GB)', flush=True)
    try:
        problem = CaseProblem(spec)
        if problem.n_presolved:
            print(f'  [{spec.name}] presolved: {problem.n_presolved} 个设计的 F/G 直接采纳（不求解）', flush=True)
        _kw = {}
        if sampling is not None:
            _kw['sampling'] = sampling
        _algo = str(getattr(spec, 'algorithm', 'nsga2') or 'nsga2').lower()
        if _algo == 'nsga3':
            algorithm = _make_nsga3(spec, _kw)
        else:
            algorithm = NSGA2(pop_size=spec.pop_size, crossover=SBX(prob=0.9, eta=10), mutation=PM(prob=0.3, eta=15), repair=IndexRound(spec.integer_vars), eliminate_duplicates=True, **_kw)
        cb_t0 = [time.time()]
        _t_start = time.time()

        def _cb(algorithm):
            gen = algorithm.n_gen
            now = time.time()
            dt = now - cb_t0[0]
            cb_t0[0] = now
            pop = algorithm.pop
            X = pop.get('X')
            if X is not None:
                metrics = _compute_metrics(spec, gen, pop, dt)
                if metrics is not None:
                    _cs = problem.cache_summary()
                    _ws = _warm_summary(problem)
                    metrics['cache_hit'] = _cs['hits']
                    metrics['cache_miss'] = _cs['misses']
                    metrics['cache_mb'] = _cs['bytes'] / 2 ** 20
                    metrics['warm_hit'] = _ws.get('hits', 0)
                    metrics['warm_miss'] = _ws.get('misses', 0)
                    save_checkpoint(spec, gen, X, pop.get('F'), pop.get('G'), pop.get('_info'))
                    append_metrics_csv(spec, metrics, header)
                    print(f"  [{spec.name}] gen {gen:>4} | feas {metrics['n_feasible']}/{spec.pop_size} | gen_t {dt:>5.1f}s | warm {_ws.get('hit_rate', 0.0) * 100:.0f}% {_ws.get('hits', 0)}/{_ws.get('hits', 0) + _ws.get('misses', 0)} ({_ws.get('bytes', 0) / 2 ** 30:.1f}/{_ws.get('cap_bytes', 0) / 2 ** 30:.1f}GB ev{_ws.get('evicted', 0)}) | rc {_cs['hit_rate'] * 100:.0f}% ({_cs['bytes'] / 2 ** 20:.0f} MB)", flush=True)
            if callback is not None:
                callback(algorithm)
            if wall_budget_s is not None and time.time() - _t_start >= wall_budget_s:
                print(f'  [{spec.name}] 墙钟预算 {wall_budget_s:.0f}s 用尽，在第 {gen} 代干净停止（用 --resume 继续）', flush=True)
                algorithm.termination.terminate()
                algorithm.termination.perc = 1.0
        if resume:
            found = latest_checkpoint(spec.checkpoint_dir, upto=spec.n_gen)
            if found is None:
                print(f'  [{spec.name}] --resume：没有可用 checkpoint，从头开始', flush=True)
            else:
                gen0, path = found
                X0, F0, G0, info0 = load_checkpoint(path)
                if len(X0) != spec.pop_size:
                    raise RuntimeError(f'checkpoint 的 pop_size={len(X0)} 与 --pop={spec.pop_size} 不一致')
                from pymoo.core.population import Population
                algorithm.setup(problem, seed=spec.seed, verbose=False, termination=get_termination('n_gen', spec.n_gen), callback=_cb, save_history=save_history)
                pop = Population.new('X', X0, 'F', F0, 'G', G0) if G0 is not None else Population.new('X', X0, 'F', F0)
                if info0 is not None:
                    pop.set('_info', info0)
                algorithm.pop = pop
                algorithm.n_gen = gen0 + 1
                algorithm.is_initialized = True
                algorithm.start_time = time.time()
                algorithm.pop = algorithm.survival.do(problem, pop, n_survive=spec.pop_size, algorithm=algorithm, random_state=algorithm.random_state)
                print(f'  [{spec.name}] 从 {os.path.basename(path)} 续跑（已有 {gen0} 代，目标 {spec.n_gen} 代）', flush=True)
        res = minimize(problem, algorithm, termination=get_termination('n_gen', spec.n_gen), seed=spec.seed, save_history=save_history, verbose=False, copy_algorithm=False, callback=_cb)
    finally:
        _POOL.close()
        _POOL.join()
        _POOL = None
    return res
