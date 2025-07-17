import cupy as cp
import cupyx as cpx
import numpy as np

# --- Load data ---
data = np.load("firedrake_data.npz")
derivs_gpu = cp.asarray(data["grad_basis"], dtype=cp.float64)  # (J, L, M)
basis_funcs_gpu = cp.asarray(data["basis"])                    # (K, J)
cg_node_map_gpu = cp.asarray(data["cg_map"])                   # (I, K)
cg_data_gpu = cp.zeros_like(data["empty_data"])                # (N,)
coord_node_map_gpu = cp.asarray(data["coords_map"])            # (I, J)
coord_data_gpu = cp.asarray(data["coords"], dtype=cp.float64)  # (N, M)
weights_gpu = cp.asarray(data["weights"])                     # (J,)

# --- Build cell-wise coordinate arrays ---
cell_coords = cp.take(coord_data_gpu, coord_node_map_gpu, axis=0)  # (I, J, M)

I, J, M = cell_coords.shape
_, L, _ = derivs_gpu.shape
K, _ = basis_funcs_gpu.shape

# --- Per-cell assembly (using CPU for-loop over I) ---
contracted = cp.zeros((I, K), dtype=cp.float64)

for i in range(I):  # I = 5000 (number of cells)
    det_jacobians = cp.zeros(L, dtype=cp.float64)
    for l in range(L):  # L = 25 (number of quadrature points)
        jac = cp.zeros((M, M), dtype=cp.float64)
        for m in range(M):  # M = 2 (number of spatial dimensions)
            for n in range(M):
                for j in range(J):  # J = 2 (number of nodes per cell)
                    jac[m, n] += cell_coords[i, j, m] * derivs_gpu[j, l, n]
        det_jacobians[l] = cp.fabs(cp.linalg.det(jac))
    for k in range(K):  # K = 6
        for l in range(L):
            contracted[i, k] += det_jacobians[l] * basis_funcs_gpu[k, l] * weights_gpu[l]

            
# --- Scatter into global vector ---
cpx.scatter_add(cg_data_gpu, cg_node_map_gpu, contracted)

# --- Check results ---
output = cg_data_gpu
print("GPU:", output)
print("Firedrake:", data["expected"])
assert np.allclose(data["expected"], output), "Mismatch!"
print("✅ Python-for loop version matches.")
1
