import time
from functools import wraps


def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        t0 = time.perf_counter_ns()
        result = f(*args, **kwargs)
        t1 = time.perf_counter_ns()
        print(f"func {f.__name__} :-> {((t1 - t0) * 1e-6):.0f} ms")
        return result

    return wrap
