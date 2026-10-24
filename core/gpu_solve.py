"""Jacobi-preconditioned conjugate gradients on the GPU for the surface completion fill.

The system is the same symmetric positive definite graph Laplacian the CPU solver (SciPy ``cg``) gets, with the
same preconditioner, tolerance and iteration limit; all colour channels are solved together. Double precision
is used because a sparse matrix-vector product is memory-bound, so it costs no more than single precision here.
"""
import warnings

import numpy as np

CHECK_EVERY = 10   # iterations between convergence checks (each check is a host synchronisation)


def solve_cg_columns_gpu(matrix, rhs, inverse_diagonal, device, rtol=1e-6, maxiter=1500):
    """Solve matrix @ x = rhs[:, k] for every column. matrix: SciPy sparse (n,n) SPD; rhs: (n,C) array."""
    import torch
    csr = matrix.tocsr()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')   # "Sparse CSR tensor support is in beta state"
        A = torch.sparse_csr_tensor(torch.from_numpy(csr.indptr.astype(np.int64)), torch.from_numpy(csr.indices.astype(np.int64)),
                                    torch.from_numpy(csr.data.astype(np.float64)), size=csr.shape, dtype=torch.float64).to(device)
    b = torch.from_numpy(np.ascontiguousarray(rhs, dtype=np.float64)).to(device)
    M = torch.from_numpy(np.asarray(inverse_diagonal, np.float64)).to(device)[:, None]
    x = torch.zeros_like(b)
    r = b.clone()
    z = M * r
    p = z.clone()
    rz = (r * z).sum(0)
    threshold = rtol * b.norm(dim=0)
    iterations = 0
    for iterations in range(1, maxiter + 1):
        Ap = A @ p
        alpha = rz / (p * Ap).sum(0).clamp_min(1e-300)
        x += alpha * p
        r -= alpha * Ap
        if iterations % CHECK_EVERY == 0 and bool((r.norm(dim=0) <= threshold).all()):
            break
        z = M * r
        rz_next = (r * z).sum(0)
        p = z + (rz_next / rz.clamp_min(1e-300)) * p
        rz = rz_next
    return x.cpu().numpy(), iterations
