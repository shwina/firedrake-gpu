"""
BASELINE VERSION - Single Thread per Element FEM Assembly

This version uses the standard approach where each CUDA thread processes
one finite element independently.

Implementation characteristics:
- Simple memory access patterns
- Individual atomic operations per thread
- Standard finite element assembly approach
"""

import cupy as cp
import cupyx as cpx
import numpy as np
import timeit

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

# --- Kernel
kernel_code = f"""
extern "C" __global__
void fem_assembly(
    const double* __restrict__ coords_data, // (N, M)
    const int* __restrict__ coords_map,     // (I, J)
    const double* __restrict__ derivs,      // (J, L, M)
    const double* __restrict__ weights,     // (L,)
    const double* __restrict__ basis,       // (K, L)
    const int* __restrict__ cg_map,         // (I, K)
    double* __restrict__ cg_data,           // (N,)
    int I, int J, int L, int M, int K, int N)
{{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= I) return;

    double det_jacobians[64]; // support up to L=64

    // For each quadrature point l, compute |det J(l)|
    for (int l=0; l<L; l++) {{
        double jac[4][4] = {{0.0}}; // up to 3D
        for (int m=0; m<M; m++) {{
            for (int n=0; n<M; n++) {{
                double sum = 0.0;
                for (int j=0; j<J; j++) {{
                    int node = coords_map[i*J + j];
                    sum += coords_data[node*M + m] * derivs[(j*L + l)*M + n];
                }}
                jac[m][n] = sum;
            }}
        }}
        // Compute |det J| for 2D
        double det = 0.0;
        if (M == 2) {{
            det = fabs(jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]);
        }} else if (M == 3) {{
            det = fabs(
                jac[0][0]*(jac[1][1]*jac[2][2] - jac[1][2]*jac[2][1])
              - jac[0][1]*(jac[1][0]*jac[2][2] - jac[1][2]*jac[2][0])
              + jac[0][2]*(jac[1][0]*jac[2][1] - jac[1][1]*jac[2][0])
            );
        }}
        det_jacobians[l] = det;
    }}

    // Integrate against basis functions and weights
    double local_contrib[32] = {{0.0}}; // up to K=32
    for (int k=0; k<K; k++) {{
        double sum = 0.0;
        for (int l=0; l<L; l++) {{
            sum += det_jacobians[l] * basis[k*L + l] * weights[l];
        }}
        local_contrib[k] = sum;
    }}

    // Scatter-add into global vector
    for (int k=0; k<K; k++) {{
        int global_idx = cg_map[i*K + k];
        atomicAdd(&cg_data[global_idx], local_contrib[k]);
    }}
}}
"""

# --- Compile kernel
fem_kernel = cp.RawKernel(kernel_code, "fem_assembly")

# --- Launch configuration
threads_per_block = 128
blocks = (I + threads_per_block - 1) // threads_per_block

print(blocks, threads_per_block)

# --- Warmup run
print("Performing warmup run...")
fem_kernel(
    (blocks,), (threads_per_block,),
    (
        coord_data_gpu.ravel(),
        coord_node_map_gpu.ravel(),
        derivs_gpu.ravel(),
        weights_gpu,
        basis_funcs_gpu.ravel(),
        cg_node_map_gpu.ravel(),
        cg_data_gpu,
        I, J, L, M, K, N
    )
)
cp.cuda.runtime.deviceSynchronize()

# Reset the output
cg_data_gpu.fill(0.0)

# --- Run assembly
print("Running assembly...")
t1 = timeit.default_timer()
fem_kernel(
    (blocks,), (threads_per_block,),
    (
        coord_data_gpu.ravel(),
        coord_node_map_gpu.ravel(),
        derivs_gpu.ravel(),
        weights_gpu,
        basis_funcs_gpu.ravel(),
        cg_node_map_gpu.ravel(),
        cg_data_gpu,
        I, J, L, M, K, N
    )
)
cp.cuda.runtime.deviceSynchronize()
t2 = timeit.default_timer()

print("Time for assembly:", t2 - t1)
# --- Check result
output = cg_data_gpu
print("GPU:", output)
print("Firedrake:", data["expected"])
assert np.allclose(data["expected"], output,
                   rtol=1e-10, atol=1e-12), "Mismatch!"
print("✅ RawKernel version matches Firedrake output.")
