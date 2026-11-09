from dataclasses import dataclass, field

@dataclass
class CaseSpec:
    name: str
    bounds: list
    n_obj: int
    n_constr: int
    var_names: list
    worker_module: str
    integer_vars: tuple = ()
    n_ax: int = 20
    table_ranges: list = field(default_factory=list)
    checkpoint_dir: str = ''
    pop_size: int = 80
    n_gen: int = 100
    n_jobs: int = 12
    seed: int = 42
    solver_kwargs: dict = field(default_factory=dict)
    memory_budget_bytes: int = 32 * 1024 ** 3
    result_cache_bytes: int = None
    state_cache_bytes: int = None
    state_cache_max_bytes: int = None
    warm_start: bool = False
    algorithm: str = 'nsga2'
    objective_tag: str = ''
    presolved: object = None

    def __post_init__(self):
        self.bounds = [tuple((float(b) for b in pair)) for pair in self.bounds]
        self.var_names = list(self.var_names)
        self.integer_vars = tuple((int(i) for i in self.integer_vars))
        if len(self.var_names) != len(self.bounds):
            raise ValueError('CaseSpec: len(var_names)=%d != len(bounds)=%d' % (len(self.var_names), len(self.bounds)))

    def cache_split(self):
        if not self.memory_budget_bytes or self.memory_budget_bytes <= 0:
            return (0, 0)
        if self.result_cache_bytes is not None:
            res = int(self.result_cache_bytes)
        else:
            n_distinct = max(1, self.pop_size * self.n_gen)
            res = min(1 << 30, n_distinct * 2 * 8 * 1024)
        if self.state_cache_bytes is not None:
            per_worker = int(self.state_cache_bytes)
        else:
            rest = max(0, int(self.memory_budget_bytes) - res)
            cap = 512 << 20 if self.state_cache_max_bytes is None else max(0, int(self.state_cache_max_bytes))
            per_worker = min(cap, rest // max(1, self.n_jobs))
        return (res, per_worker)
