import importlib
_MOD = None
_STATES = None
_SIG = None
_SOLVER_KW = {}
_COST_FN = None

def init_worker(module_name: str, state_cache_bytes: int=0, solver_kwargs=None):
    global _MOD, _STATES, _SIG, _SOLVER_KW, _COST_FN
    _SOLVER_KW = dict(solver_kwargs or {})
    _MOD = importlib.import_module(module_name)
    _COST_FN = getattr(_MOD, 'cost_hint', None)
    _SIG = getattr(_MOD, 'warm_signature', None)
    if state_cache_bytes and _SIG is not None:
        from pyht.optim.cache import StateCache
        _STATES = StateCache(state_cache_bytes)
    else:
        _STATES = None
    if hasattr(_MOD, 'init_worker'):
        _MOD.init_worker()

def warm_stats():
    if _STATES is None:
        return {'enabled': False}
    out = _STATES.stats()
    out['enabled'] = True
    return out

def eval_warm(payload):
    x, state = payload
    last_state = getattr(_MOD, 'last_state', None)
    try:
        res = _MOD.evaluate(x, warm_start=state, solver_opts=_SOLVER_KW or None)
    except TypeError:
        return (_MOD.evaluate(x), None)
    info = res[2] if len(res) > 2 else None
    if not callable(last_state) or not info or info.get('solver_fail'):
        return (res, None)
    return (res, last_state())

def cost_hint(x):
    if _COST_FN is None:
        return 0.0
    try:
        return float(_COST_FN(x))
    except Exception:
        return 0.0

def eval_generic(x):
    return _MOD.evaluate(x, solver_opts=_SOLVER_KW or None)

def eval_batch(xs):
    if _STATES is None or _SIG is None:
        return [_MOD.evaluate(x) for x in xs]
    last_state = getattr(_MOD, 'last_state', None)
    out = []
    for x in xs:
        try:
            sig = _SIG(x)
        except Exception:
            sig = None
        state = _STATES.get(sig) if sig is not None else None
        try:
            res = _MOD.evaluate(x, warm_start=state)
        except TypeError:
            globals()['_STATES'] = None
            return [_MOD.evaluate(y) for y in xs]
        out.append(res)
        if sig is None or not callable(last_state):
            continue
        info = res[2] if len(res) > 2 else None
        if not info or info.get('solver_fail'):
            continue
        sv = last_state()
        if sv is not None:
            try:
                _STATES.put(sig, sv)
            except Exception:
                pass
    return out
