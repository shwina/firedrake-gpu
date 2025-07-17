"""
OPTIMIZED VERSION - 4-Thread Subwarp Cooperative FEM Assembly

This version uses 4-thread subwarps for cooperative finite element assembly.
Key optimizations:
- Subwarp cooperation: 4 threads work together on each element
- Shared memory caching of weights array
- Warp shuffle communication for sharing Jacobian determinants
- Reduced atomic contention: manual reduction before atomicAdd
- Improved memory access patterns

To run: python minimal-raw-kernel-subwarp.py
"""

import cupy as cp
import cupyx as cpx
import numpy as np

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
    // Use 4-thread subwarps for cooperation
    const int SUBWARP_SIZE = 4;
    
    // Each subwarp handles one element
    int element_id = (blockDim.x * blockIdx.x + threadIdx.x) / SUBWARP_SIZE;
    int lane_id = threadIdx.x % SUBWARP_SIZE;
    
    if (element_id >= I) return;
    
    // Shared memory for caching weights (better memory access)
    __shared__ double shared_weights[64];  // up to L=64
    
    // Cooperatively load weights into shared memory
    for (int i = threadIdx.x; i < L; i += blockDim.x) {{
        shared_weights[i] = weights[i];
    }}
    __syncthreads();
    
    double det_jacobians[64]; // support up to L=64
    
    // Compute Jacobian determinants - distribute quadrature points across subwarp
    for (int l = lane_id; l < L; l += SUBWARP_SIZE) {{
        double jac[4][4] = {{0.0}}; // up to 3D
        
        for (int m = 0; m < M; m++) {{
            for (int n = 0; n < M; n++) {{
                double sum = 0.0;
                for (int j = 0; j < J; j++) {{
                    int node = coords_map[element_id * J + j];
                    sum += coords_data[node * M + m] * derivs[(j * L + l) * M + n];
                }}
                jac[m][n] = sum;
            }}
        }}
        
        // Compute |det J| for 2D/3D
        double det = 0.0;
        if (M == 2) {{
            det = fabs(jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]);
        }} else if (M == 3) {{
            det = fabs(
                jac[0][0] * (jac[1][1] * jac[2][2] - jac[1][2] * jac[2][1])
              - jac[0][1] * (jac[1][0] * jac[2][2] - jac[1][2] * jac[2][0])
              + jac[0][2] * (jac[1][0] * jac[2][1] - jac[1][1] * jac[2][0])
            );
        }}
        det_jacobians[l] = det;
    }}
    
    // Share Jacobian determinants across subwarp using simple warp shuffles
    for (int l = 0; l < L; l++) {{
        int source_lane = l % SUBWARP_SIZE;
        int source_thread = (threadIdx.x / SUBWARP_SIZE) * SUBWARP_SIZE + source_lane;
        det_jacobians[l] = __shfl_sync(0xffffffff, det_jacobians[l], source_thread);
    }}
    
    // Compute basis function contributions - distribute across subwarp
    double local_contrib[32] = {{0.0}}; // up to K=32
    
    for (int k = lane_id; k < K; k += SUBWARP_SIZE) {{
        double sum = 0.0;
        for (int l = 0; l < L; l++) {{
            sum += det_jacobians[l] * basis[k * L + l] * shared_weights[l];
        }}
        local_contrib[k] = sum;
    }}
    
    // Scatter-add with reduced atomic contention using manual reduction
    for (int k = 0; k < K; k++) {{
        double contribution = (k % SUBWARP_SIZE == lane_id) ? local_contrib[k] : 0.0;
        
        // Manual 4-thread reduction using shuffles
        contribution += __shfl_down_sync(0xffffffff, contribution, 2);
        contribution += __shfl_down_sync(0xffffffff, contribution, 1);
        
        // Only thread 0 of each subwarp does atomic add
        if (lane_id == 0 && contribution != 0.0) {{
            int global_idx = cg_map[element_id * K + k];
            atomicAdd(&cg_data[global_idx], contribution);
        }}
    }}
}}
"""

# --- Compile kernel
fem_kernel = cp.RawKernel(kernel_code, "fem_assembly")

# --- Launch configuration  
# Since each subwarp (4 threads) handles one element, we need 4x more threads
SUBWARP_SIZE = 4
threads_per_block = 128
elements_per_block = threads_per_block // SUBWARP_SIZE  # 32 elements per block
blocks = (I + elements_per_block - 1) // elements_per_block

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
