"""Allow native CPU math libraries to use every available logical processor.

This controls existing BLAS/OpenMP pools, without copying meshes into worker
processes. Algorithms without parallel implementations still execute serially.
"""
import os
import threading

_LOCK = threading.RLock()
_CONTROLLER = None


def parallel_workers():
    """Rollback affects independent CPU jobs, not native math pool settings."""
    if os.environ.get('UTX_CPU_PARALLEL', '1') == '0':
        return 1
    return getattr(os, 'process_cpu_count', os.cpu_count)() or 1


def solve_cg_columns(matrix, rhs, preconditioner):
    """Solve independent colour channels against one shared sparse matrix.

    Each channel keeps the original solver and stopping tolerance. Limit nested
    BLAS pools during these jobs to avoid threads multiplying each other.
    """
    from concurrent.futures import ThreadPoolExecutor
    from scipy.sparse.linalg import cg
    from threadpoolctl import threadpool_limits
    import numpy as np

    workers = min(parallel_workers(), rhs.shape[1])
    def solve(channel):
        return cg(matrix, rhs[:, channel], rtol=1e-6, maxiter=1500,
                  M=preconditioner)[0]
    with threadpool_limits(limits=1, user_api='blas'):
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                columns = list(pool.map(solve, range(rhs.shape[1])))
        else:
            columns = [solve(i) for i in range(rhs.shape[1])]
    return np.column_stack(columns), workers


def configure_cpu():
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits

    global _CONTROLLER
    with _LOCK:
        count = (getattr(os, "process_cpu_count", os.cpu_count)() or 1)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[key] = str(count)
        if torch.get_num_threads() != count:
            torch.set_num_threads(count)
        # Reapply for libraries imported since startup (for example SciPy).
        _CONTROLLER = threadpool_limits(limits=count)
        return {"logical_processors": count, "torch_threads": torch.get_num_threads(),
                "native_pools": [{k: pool.get(k) for k in
                                  ("internal_api", "prefix", "num_threads")}
                                 for pool in threadpool_info()]}
