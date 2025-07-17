"""
NUMBA CUDA VERSION - Single Thread per Element FEM Assembly

This version uses Numba's CUDA JIT compiler instead of CuPy's RawKernel
for the CUDA kernel implementation.

Implementation characteristics:
- Same memory access patterns as RawKernel version
- Individual atomic operations per thread
- Standard finite element assembly approach
- Python-based kernel with Numba CUDA decorators
"""

import cupy as cp
import numpy as np
import timeit
import math
from numba import cuda
from numba import float64

# --- Load data
data = np.load("firedrake_data.npz")
derivs_gpu = cp.asarray(data["grad_basis"], dtype=cp.float64)  # (J, L, M)
basis_funcs_gpu = cp.asarray(data["basis"])                    # (K, L)
cg_node_map_gpu = cp.asarray(data["cg_map"])                   # (I, K)
cg_data_gpu = cp.zeros_like(data["empty_data"])                # (N,)
coord_node_map_gpu = cp.asarray(data["coords_map"])            # (I, J)
coord_data_gpu = cp.asarray(data["coords"], dtype=cp.float64)  # (N, M)
weights_gpu = cp.asarray(data["weights"])                     # (L,)

# --- Dimensions
I, J, M = coord_node_map_gpu.shape[0], derivs_gpu.shape[0], coord_data_gpu.shape[1]
L = derivs_gpu.shape[1]
K = basis_funcs_gpu.shape[0]
N = cg_data_gpu.shape[0]

# --- Numba CUDA Kernel (Optimized)
@cuda.jit
def fem_assembly_numba(coords_data, coords_map, derivs, weights, basis, cg_map, cg_data, I, J, L, M, K, N):
    i = cuda.grid(1)
    if i >= I:
        return
    
    # Use smaller, register-friendly local arrays
    det_jacobians = cuda.local.array(64, dtype=float64)  # support up to L=64
    local_contrib = cuda.local.array(32, dtype=float64)  # support up to K=32
    
    # For each quadrature point l, compute |det J(l)|
    for l in range(L):
        # Use individual variables instead of arrays for jacobian (register-friendly)
        jac00 = 0.0; jac01 = 0.0; jac02 = 0.0; jac03 = 0.0
        jac10 = 0.0; jac11 = 0.0; jac12 = 0.0; jac13 = 0.0  
        jac20 = 0.0; jac21 = 0.0; jac22 = 0.0; jac23 = 0.0
        jac30 = 0.0; jac31 = 0.0; jac32 = 0.0; jac33 = 0.0
        
        # Compute jacobian elements directly
        for j in range(J):
            node = coords_map[i * J + j]
            deriv_offset = (j * L + l) * M
            
            if M >= 1:
                coord0 = coords_data[node * M + 0]
                jac00 += coord0 * derivs[deriv_offset + 0]
                if M >= 2:
                    jac01 += coord0 * derivs[deriv_offset + 1]
                    if M >= 3:
                        jac02 += coord0 * derivs[deriv_offset + 2]
            
            if M >= 2:
                coord1 = coords_data[node * M + 1]
                jac10 += coord1 * derivs[deriv_offset + 0]
                jac11 += coord1 * derivs[deriv_offset + 1]
                if M >= 3:
                    jac12 += coord1 * derivs[deriv_offset + 2]
            
            if M >= 3:
                coord2 = coords_data[node * M + 2]
                jac20 += coord2 * derivs[deriv_offset + 0]
                jac21 += coord2 * derivs[deriv_offset + 1]
                jac22 += coord2 * derivs[deriv_offset + 2]
        
        # Compute |det J| for 2D or 3D
        det = 0.0
        if M == 2:
            det = abs(jac00 * jac11 - jac01 * jac10)
        elif M == 3:
            det = abs(
                jac00 * (jac11 * jac22 - jac12 * jac21)
                - jac01 * (jac10 * jac22 - jac12 * jac20)
                + jac02 * (jac10 * jac21 - jac11 * jac20)
            )
        det_jacobians[l] = det
    
    # Integrate against basis functions and weights
    for k in range(K):
        sum_val = 0.0
        for l in range(L):
            sum_val += det_jacobians[l] * basis[k * L + l] * weights[l]
        local_contrib[k] = sum_val
    
    # Scatter-add into global vector
    for k in range(K):
        global_idx = cg_map[i * K + k]
        cuda.atomic.add(cg_data, global_idx, local_contrib[k])

# --- Launch configuration
threads_per_block = 128
blocks = (I + threads_per_block - 1) // threads_per_block

print(blocks, threads_per_block)

# --- Warmup run
print("Performing warmup run...")
fem_assembly_numba[blocks, threads_per_block](
    coord_data_gpu.ravel(),
    coord_node_map_gpu.ravel(),
    derivs_gpu.ravel(),
    weights_gpu,
    basis_funcs_gpu.ravel(),
    cg_node_map_gpu.ravel(),
    cg_data_gpu,
    I, J, L, M, K, N
)
cuda.synchronize()

# Reset the output
cg_data_gpu.fill(0.0)

# --- Run assembly
print("Running assembly...")
t1 = timeit.default_timer()
fem_assembly_numba[blocks, threads_per_block](
    coord_data_gpu.ravel(),
    coord_node_map_gpu.ravel(),
    derivs_gpu.ravel(),
    weights_gpu,
    basis_funcs_gpu.ravel(),
    cg_node_map_gpu.ravel(),
    cg_data_gpu,
    I, J, L, M, K, N
)
cuda.synchronize()
t2 = timeit.default_timer()

print("Time for assembly:", t2 - t1)
# --- Check result
output = cg_data_gpu
print("GPU:", output)
print("Firedrake:", data["expected"])
assert np.allclose(data["expected"], output,
                   rtol=1e-10, atol=1e-12), "Mismatch!"
print("✅ Numba CUDA version matches Firedrake output.") 