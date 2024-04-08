#include <bits/stdc++.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector_types.h>
#include <cuda_fp16.h>

using namespace nvcuda;
using namespace std;

#define CP_ASYNC_CA(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))


#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)

__global__ void flash10_async_q_inregs_16x64x160(half* Q, half* K, half* V, half* O, int s){
    
    Q += blockIdx.z * s * 64 + blockIdx.x * 64*64;
    O += blockIdx.z * s * 64 + blockIdx.x * 64*64;
    K += blockIdx.z * s * 64;
    V += blockIdx.z * s * 64;

    __shared__ half k_sh[160*72];
    __shared__ half v_sh[160*72];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag[4];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qk_exp_frag[1];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> qk_frag[10];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> qkv_frag[4];

    int wid = (threadIdx.x / 32);
    Q += wid*16*64;
    O += wid*16*64;

    const half scale = 1.f/sqrtf(64.f);
    
    #pragma unroll
    for(int i = 0; i < 4; i++){
        wmma::load_matrix_sync(q_frag[i], Q + (i/4)*16*64 + (i%4)*16, 64);
        wmma::fill_fragment(qkv_frag[i], 0.0f);
        
        #pragma unroll
        for(int j = 0; j < q_frag[i].num_elements; j++){
            q_frag[i].x[j] *= scale;
        }
    }

    half2 a_running_max = __float2half2_rn(0.f);
    half2 a_sum = __float2half2_rn(0.f);

    for(int o = 0; o < s; o += 160){
        for(int i = threadIdx.x; i < 160 * 64 / 8; i += blockDim.x){
            int j = i + i/8;
            uint32_t smem = __cvta_generic_to_shared(reinterpret_cast<float4*>(k_sh)+j);
            CP_ASYNC_CG(smem, reinterpret_cast<float4*>(K) + i, 16);
            
            smem = __cvta_generic_to_shared(reinterpret_cast<float4*>(v_sh)+j);
            CP_ASYNC_CG(smem, reinterpret_cast<float4*>(V) + i, 16);
        }
        
        CP_ASYNC_WAIT_ALL();
        __syncthreads();
        
        K += 160*64; V += 160*64;
        half2 a_max = a_running_max;

        #pragma unroll
        for(int i = 0; i < 10; i ++){
            wmma::fill_fragment(qk_frag[i], 0.0f);
            #pragma unroll
            for(int j = 0; j < 4; j++){
                wmma::load_matrix_sync(k_frag, k_sh + i*16*72 + j*16, 72);
                wmma::mma_sync(qk_frag[i], q_frag[j], k_frag, qk_frag[i]);
            }
        }
        __syncwarp();
        #pragma unroll
        for(int i = 0; i < 10; i++){
            a_max.x = max(a_max.x, qk_frag[i].x[0]);
            a_max.x = max(a_max.x, qk_frag[i].x[1]);
            a_max.y = max(a_max.y, qk_frag[i].x[2]);
            a_max.y = max(a_max.y, qk_frag[i].x[3]);
            a_max.x = max(a_max.x, qk_frag[i].x[4]);
            a_max.x = max(a_max.x, qk_frag[i].x[5]);
            a_max.y = max(a_max.y, qk_frag[i].x[6]);
            a_max.y = max(a_max.y, qk_frag[i].x[7]);
        }

        #pragma unroll
        for(int j = 2; j > 0; j >>= 1){
            a_max.x = max(a_max.x, __shfl_xor_sync(uint32_t(-1), a_max.x, j));
            a_max.y = max(a_max.y, __shfl_xor_sync(uint32_t(-1), a_max.y, j));
        }

        half rescale_x = exp2f(__half2float(a_running_max.x-a_max.x));
        half rescale_y = exp2f(__half2float(a_running_max.y-a_max.y));
        
        a_sum.x *= rescale_x;
        a_sum.y *= rescale_y;
        
        #pragma unroll
        for(int i = 0; i < 4; i ++){
            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qkv_frag[i].x[j] *= rescale_x;
                qkv_frag[i].x[j+1] *= rescale_x;
                qkv_frag[i].x[j+2] *= rescale_y;
                qkv_frag[i].x[j+3] *= rescale_y;
            }
        }

        #pragma unroll
        for(int i = 0; i < 10; i++){
            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qk_exp_frag[0].x[j] = qk_exp_frag[0].x[j+8] = exp2f(__half2float(qk_frag[i].x[j] - a_max.x));
                qk_exp_frag[0].x[j+1] = qk_exp_frag[0].x[j+9] = exp2f(__half2float(qk_frag[i].x[j+1] - a_max.x));
                qk_exp_frag[0].x[j+2] = qk_exp_frag[0].x[j+10] = exp2f(__half2float(qk_frag[i].x[j+2] - a_max.y));
                qk_exp_frag[0].x[j+3] = qk_exp_frag[0].x[j+11] = exp2f(__half2float(qk_frag[i].x[j+3] - a_max.y));
            }

            #pragma unroll
            for(int j = 0; j < 4; j++){
                wmma::load_matrix_sync(v_frag, v_sh + i*16*72 + j*16, 72);
                wmma::mma_sync(qkv_frag[j], qk_exp_frag[0], v_frag, qkv_frag[j]);
            }

            a_sum.x += qk_exp_frag[0].x[0] + qk_exp_frag[0].x[1];
            a_sum.y += qk_exp_frag[0].x[2] + qk_exp_frag[0].x[3];
            a_sum.x += qk_exp_frag[0].x[4] + qk_exp_frag[0].x[5];
            a_sum.y += qk_exp_frag[0].x[6] + qk_exp_frag[0].x[7];
        }
        a_running_max = a_max;
    }
    #pragma unroll
    for(int j = 2; j > 0; j >>= 1){
        a_sum += __shfl_xor_sync(uint32_t(-1), a_sum, j);
    }
    #pragma unroll
    for(int i = 0; i < 4; i++){
        qkv_frag[i].x[0] /= a_sum.x;
        qkv_frag[i].x[1] /= a_sum.x;
        qkv_frag[i].x[2] /= a_sum.y;
        qkv_frag[i].x[3] /= a_sum.y;
        qkv_frag[i].x[4] /= a_sum.x;
        qkv_frag[i].x[5] /= a_sum.x;
        qkv_frag[i].x[6] /= a_sum.y;
        qkv_frag[i].x[7] /= a_sum.y;
        wmma::store_matrix_sync(O + i*16, qkv_frag[i], 64, wmma::mem_row_major);
    }
}

__global__ void flash10_async_q_inregs_32x64x160(half* Q, half* K, half* V, half* O, int s){
    
    Q += blockIdx.z * s * 64 + blockIdx.x * 128*64;
    O += blockIdx.z * s * 64 + blockIdx.x * 128*64;
    K += blockIdx.z * s * 64;
    V += blockIdx.z * s * 64;

    __shared__ half k_sh[160*72];
    __shared__ half v_sh[160*72];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag[8];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qk_exp_frag[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> qk_frag[20];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> qkv_frag[8];

    int wid = (threadIdx.x / 32);
    Q += 2*wid*16*64;
    O += 2*wid*16*64;

    const half scale = 1.f/sqrtf(64.f);
    
    #pragma unroll
    for(int i = 0; i < 8; i++){
        wmma::load_matrix_sync(q_frag[i], Q + (i/4)*16*64 + (i%4)*16, 64);
        wmma::fill_fragment(qkv_frag[i], 0.0f);
        
        #pragma unroll
        for(int j = 0; j < q_frag[i].num_elements; j++){
            q_frag[i].x[j] *= scale;
        }
    }

    half2 a_running_max = __float2half2_rn(0.f);
    half2 b_running_max = __float2half2_rn(0.f);
    half2 a_sum = __float2half2_rn(0.f);
    half2 b_sum = __float2half2_rn(0.f);

    for(int o = 0; o < s; o += 160){
        for(int i = threadIdx.x; i < 160 * 64 / 8; i += blockDim.x){
            int j = i + i/8;
            uint32_t smem = __cvta_generic_to_shared(reinterpret_cast<float4*>(k_sh)+j);
            CP_ASYNC_CG(smem, reinterpret_cast<float4*>(K) + i, 16);
            
            smem = __cvta_generic_to_shared(reinterpret_cast<float4*>(v_sh)+j);
            CP_ASYNC_CG(smem, reinterpret_cast<float4*>(V) + i, 16);
        }
        
        CP_ASYNC_WAIT_ALL();
        __syncthreads();
        
        K += 160*64; V += 160*64;
        half2 a_max = a_running_max;
        half2 b_max = b_running_max;

        #pragma unroll
        for(int i = 0; i < 10; i ++){
            wmma::fill_fragment(qk_frag[i], 0.0f);
            wmma::fill_fragment(qk_frag[i+10], 0.0f);
            #pragma unroll
            for(int j = 0; j < 4; j++){
                wmma::load_matrix_sync(k_frag, k_sh + i*16*72 + j*16, 72);
                wmma::mma_sync(qk_frag[i], q_frag[j], k_frag, qk_frag[i]);
                wmma::mma_sync(qk_frag[i+10], q_frag[j+4], k_frag, qk_frag[i+10]);
            }
        }
        __syncwarp();
        #pragma unroll
        for(int i = 0; i < 10; i++){
            a_max.x = max(a_max.x, qk_frag[i].x[0]);
            a_max.x = max(a_max.x, qk_frag[i].x[1]);
            a_max.y = max(a_max.y, qk_frag[i].x[2]);
            a_max.y = max(a_max.y, qk_frag[i].x[3]);
            a_max.x = max(a_max.x, qk_frag[i].x[4]);
            a_max.x = max(a_max.x, qk_frag[i].x[5]);
            a_max.y = max(a_max.y, qk_frag[i].x[6]);
            a_max.y = max(a_max.y, qk_frag[i].x[7]);

            b_max.x = max(b_max.x, qk_frag[i+10].x[0]);
            b_max.x = max(b_max.x, qk_frag[i+10].x[1]);
            b_max.y = max(b_max.y, qk_frag[i+10].x[2]);
            b_max.y = max(b_max.y, qk_frag[i+10].x[3]);
            b_max.x = max(b_max.x, qk_frag[i+10].x[4]);
            b_max.x = max(b_max.x, qk_frag[i+10].x[5]);
            b_max.y = max(b_max.y, qk_frag[i+10].x[6]);
            b_max.y = max(b_max.y, qk_frag[i+10].x[7]);
        }

        #pragma unroll
        for(int j = 2; j > 0; j >>= 1){
            a_max.x = max(a_max.x, __shfl_xor_sync(uint32_t(-1), a_max.x, j));
            a_max.y = max(a_max.y, __shfl_xor_sync(uint32_t(-1), a_max.y, j));
            b_max.x = max(b_max.x, __shfl_xor_sync(uint32_t(-1), b_max.x, j));
            b_max.y = max(b_max.y, __shfl_xor_sync(uint32_t(-1), b_max.y, j));
        }

        half rescale_x = exp2f(__half2float(a_running_max.x-a_max.x));
        half rescale_y = exp2f(__half2float(a_running_max.y-a_max.y));
        
        a_sum.x *= rescale_x;
        a_sum.y *= rescale_y;
        
        #pragma unroll
        for(int i = 0; i < 4; i ++){
            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qkv_frag[i].x[j] *= rescale_x;
                qkv_frag[i].x[j+1] *= rescale_x;
                qkv_frag[i].x[j+2] *= rescale_y;
                qkv_frag[i].x[j+3] *= rescale_y;
            }
        }
        rescale_x = exp2f(__half2float(b_running_max.x-b_max.x));
        rescale_y = exp2f(__half2float(b_running_max.y-b_max.y));
        
        b_sum.x *= rescale_x;
        b_sum.y *= rescale_y;
         
        #pragma unroll
        for(int i = 4; i < 8; i ++){
            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qkv_frag[i].x[j] *= rescale_x;
                qkv_frag[i].x[j+1] *= rescale_x;
                qkv_frag[i].x[j+2] *= rescale_y;
                qkv_frag[i].x[j+3] *= rescale_y;
            }
        }

        #pragma unroll
        for(int i = 0; i < 10; i++){
            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qk_exp_frag[0].x[j] = qk_exp_frag[0].x[j+8] = exp2f(__half2float(qk_frag[i].x[j] - a_max.x));
                qk_exp_frag[0].x[j+1] = qk_exp_frag[0].x[j+9] = exp2f(__half2float(qk_frag[i].x[j+1] - a_max.x));
                qk_exp_frag[0].x[j+2] = qk_exp_frag[0].x[j+10] = exp2f(__half2float(qk_frag[i].x[j+2] - a_max.y));
                qk_exp_frag[0].x[j+3] = qk_exp_frag[0].x[j+11] = exp2f(__half2float(qk_frag[i].x[j+3] - a_max.y));
            }
            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qk_exp_frag[1].x[j] = qk_exp_frag[1].x[j+8] = exp2f(__half2float(qk_frag[i+10].x[j] - b_max.x));
                qk_exp_frag[1].x[j+1] = qk_exp_frag[1].x[j+9] = exp2f(__half2float(qk_frag[i+10].x[j+1] - b_max.x));
                qk_exp_frag[1].x[j+2] = qk_exp_frag[1].x[j+10] = exp2f(__half2float(qk_frag[i+10].x[j+2] - b_max.y));
                qk_exp_frag[1].x[j+3] = qk_exp_frag[1].x[j+11] = exp2f(__half2float(qk_frag[i+10].x[j+3] - b_max.y));
            }

            #pragma unroll
            for(int j = 0; j < 4; j++){
                wmma::load_matrix_sync(v_frag, v_sh + i*16*72 + j*16, 72);
                wmma::mma_sync(qkv_frag[j], qk_exp_frag[0], v_frag, qkv_frag[j]);
                wmma::mma_sync(qkv_frag[j+4], qk_exp_frag[1], v_frag, qkv_frag[j+4]);
            }

            a_sum.x += qk_exp_frag[0].x[0] + qk_exp_frag[0].x[1];
            a_sum.y += qk_exp_frag[0].x[2] + qk_exp_frag[0].x[3];
            a_sum.x += qk_exp_frag[0].x[4] + qk_exp_frag[0].x[5];
            a_sum.y += qk_exp_frag[0].x[6] + qk_exp_frag[0].x[7];

            b_sum.x += qk_exp_frag[1].x[0] + qk_exp_frag[1].x[1];
            b_sum.y += qk_exp_frag[1].x[2] + qk_exp_frag[1].x[3];
            b_sum.x += qk_exp_frag[1].x[4] + qk_exp_frag[1].x[5];
            b_sum.y += qk_exp_frag[1].x[6] + qk_exp_frag[1].x[7];
        }
        a_running_max = a_max;
        b_running_max = b_max;
    }
    #pragma unroll
    for(int j = 2; j > 0; j >>= 1){
        a_sum += __shfl_xor_sync(uint32_t(-1), a_sum, j);
        b_sum += __shfl_xor_sync(uint32_t(-1), b_sum, j);
    }
    #pragma unroll
    for(int i = 0; i < 4; i++){
        qkv_frag[i].x[0] /= a_sum.x;
        qkv_frag[i].x[1] /= a_sum.x;
        qkv_frag[i].x[2] /= a_sum.y;
        qkv_frag[i].x[3] /= a_sum.y;
        qkv_frag[i].x[4] /= a_sum.x;
        qkv_frag[i].x[5] /= a_sum.x;
        qkv_frag[i].x[6] /= a_sum.y;
        qkv_frag[i].x[7] /= a_sum.y;
        wmma::store_matrix_sync(O + i*16, qkv_frag[i], 64, wmma::mem_row_major);
    }
    O += 16*64;
    #pragma unroll
    for(int i = 4; i < 8; i++){
        qkv_frag[i].x[0] /= b_sum.x;
        qkv_frag[i].x[1] /= b_sum.x;
        qkv_frag[i].x[2] /= b_sum.y;
        qkv_frag[i].x[3] /= b_sum.y;
        qkv_frag[i].x[4] /= b_sum.x;
        qkv_frag[i].x[5] /= b_sum.x;
        qkv_frag[i].x[6] /= b_sum.y;
        qkv_frag[i].x[7] /= b_sum.y;
        wmma::store_matrix_sync(O + (i-4)*16, qkv_frag[i], 64, wmma::mem_row_major);
    }
}

__global__ void flash10_async_q_inregs_48x64x160(half* Q, half* K, half* V, half* O, int s){
    
    Q += blockIdx.z * s * 64 + blockIdx.x * 192*64;
    O += blockIdx.z * s * 64 + blockIdx.x * 192*64;
    K += blockIdx.z * s * 64;
    V += blockIdx.z * s * 64;

    __shared__ half k_sh[160*72];
    __shared__ half v_sh[160*72];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag[12];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qk_exp_frag[3];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> qk_frag[30];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> qkv_frag[12];

    int wid = (threadIdx.x / 32);
    Q += 3*wid*16*64;
    O += 3*wid*16*64;

    const half scale = 1.f/sqrtf(64.f);
    
    #pragma unroll
    for(int i = 0; i < 12; i++){
        wmma::load_matrix_sync(q_frag[i], Q + (i/4)*16*64 + (i%4)*16, 64);
        wmma::fill_fragment(qkv_frag[i], 0.0f);
        
        #pragma unroll
        for(int j = 0; j < q_frag[i].num_elements; j++){
            q_frag[i].x[j] *= scale;
        }
    }

    half2 a_running_max = __float2half2_rn(0.f);
    half2 b_running_max = __float2half2_rn(0.f);
    half2 c_running_max = __float2half2_rn(0.f);
    
    half2 a_sum = __float2half2_rn(0.f);
    half2 b_sum = __float2half2_rn(0.f);
    half2 c_sum = __float2half2_rn(0.f);


    for(int o = 0; o < s; o += 160){
        for(int i = threadIdx.x; i < 160 * 64 / 8; i += blockDim.x){
            int j = i + i/8;
            uint32_t smem = __cvta_generic_to_shared(reinterpret_cast<float4*>(k_sh)+j);
            CP_ASYNC_CG(smem, reinterpret_cast<float4*>(K) + i, 16);
            
            smem = __cvta_generic_to_shared(reinterpret_cast<float4*>(v_sh)+j);
            CP_ASYNC_CG(smem, reinterpret_cast<float4*>(V) + i, 16);
        }
        
        CP_ASYNC_WAIT_ALL();
        __syncthreads();
        
        K += 160*64; V += 160*64;
        half2 a_max = a_running_max;
        half2 b_max = b_running_max;
        half2 c_max = c_running_max;

        #pragma unroll
        for(int i = 0; i < 10; i++){
            wmma::fill_fragment(qk_frag[i], 0.f);
            wmma::fill_fragment(qk_frag[i+10], 0.f);
            wmma::fill_fragment(qk_frag[i+20], 0.f);
            
            #pragma unroll
            for(int j = 0; j < 4; j++){
                wmma::load_matrix_sync(k_frag, k_sh + i*16*72 + j*16, 72);
                wmma::mma_sync(qk_frag[i], q_frag[j], k_frag, qk_frag[i]);
                wmma::mma_sync(qk_frag[i+10], q_frag[j+4], k_frag, qk_frag[i+10]);
                wmma::mma_sync(qk_frag[i+20], q_frag[j+8], k_frag, qk_frag[i+20]);
            }
        }

        __syncwarp();
        #pragma unroll
        for(int i = 0; i < 10; i++){
            a_max.x = max(a_max.x, qk_frag[i].x[0]);
            a_max.x = max(a_max.x, qk_frag[i].x[1]);
            a_max.y = max(a_max.y, qk_frag[i].x[2]);
            a_max.y = max(a_max.y, qk_frag[i].x[3]);
            a_max.x = max(a_max.x, qk_frag[i].x[4]);
            a_max.x = max(a_max.x, qk_frag[i].x[5]);
            a_max.y = max(a_max.y, qk_frag[i].x[6]);
            a_max.y = max(a_max.y, qk_frag[i].x[7]);

            b_max.x = max(b_max.x, qk_frag[i+10].x[0]);
            b_max.x = max(b_max.x, qk_frag[i+10].x[1]);
            b_max.y = max(b_max.y, qk_frag[i+10].x[2]);
            b_max.y = max(b_max.y, qk_frag[i+10].x[3]);
            b_max.x = max(b_max.x, qk_frag[i+10].x[4]);
            b_max.x = max(b_max.x, qk_frag[i+10].x[5]);
            b_max.y = max(b_max.y, qk_frag[i+10].x[6]);
            b_max.y = max(b_max.y, qk_frag[i+10].x[7]);

            c_max.x = max(c_max.x, qk_frag[i+20].x[0]);
            c_max.x = max(c_max.x, qk_frag[i+20].x[1]);
            c_max.y = max(c_max.y, qk_frag[i+20].x[2]);
            c_max.y = max(c_max.y, qk_frag[i+20].x[3]);
            c_max.x = max(c_max.x, qk_frag[i+20].x[4]);
            c_max.x = max(c_max.x, qk_frag[i+20].x[5]);
            c_max.y = max(c_max.y, qk_frag[i+20].x[6]);
            c_max.y = max(c_max.y, qk_frag[i+20].x[7]);
        }

        #pragma unroll
        for(int j = 2; j > 0; j >>= 1){
            a_max.x = max(a_max.x, __shfl_xor_sync(uint32_t(-1), a_max.x, j));
            a_max.y = max(a_max.y, __shfl_xor_sync(uint32_t(-1), a_max.y, j));
            b_max.x = max(b_max.x, __shfl_xor_sync(uint32_t(-1), b_max.x, j));
            b_max.y = max(b_max.y, __shfl_xor_sync(uint32_t(-1), b_max.y, j));
            c_max.x = max(c_max.x, __shfl_xor_sync(uint32_t(-1), c_max.x, j));
            c_max.y = max(c_max.y, __shfl_xor_sync(uint32_t(-1), c_max.y, j));
        }

        half rescale_x = exp2f(__half2float(a_running_max.x-a_max.x));
        half rescale_y = exp2f(__half2float(a_running_max.y-a_max.y));
        
        a_sum.x *= rescale_x;
        a_sum.y *= rescale_y;
        
        #pragma unroll
        for(int i = 0; i < 4; i ++){
            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qkv_frag[i].x[j] *= rescale_x;
                qkv_frag[i].x[j+1] *= rescale_x;
                qkv_frag[i].x[j+2] *= rescale_y;
                qkv_frag[i].x[j+3] *= rescale_y;
            }
        }
        rescale_x = exp2f(__half2float(b_running_max.x-b_max.x));
        rescale_y = exp2f(__half2float(b_running_max.y-b_max.y));
        
        b_sum.x *= rescale_x;
        b_sum.y *= rescale_y;
         
        #pragma unroll
        for(int i = 4; i < 8; i ++){
            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qkv_frag[i].x[j] *= rescale_x;
                qkv_frag[i].x[j+1] *= rescale_x;
                qkv_frag[i].x[j+2] *= rescale_y;
                qkv_frag[i].x[j+3] *= rescale_y;
            }
        }

        rescale_x = exp2f(__half2float(c_running_max.x-c_max.x));
        rescale_y = exp2f(__half2float(c_running_max.y-c_max.y));
        
        c_sum.x *= rescale_x;
        c_sum.y *= rescale_y;
         
        #pragma unroll
        for(int i = 8; i < 12; i ++){
            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qkv_frag[i].x[j] *= rescale_x;
                qkv_frag[i].x[j+1] *= rescale_x;
                qkv_frag[i].x[j+2] *= rescale_y;
                qkv_frag[i].x[j+3] *= rescale_y;
            }
        }
        #pragma unroll
        for(int i = 0; i < 10; i++){
            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qk_exp_frag[0].x[j] = qk_exp_frag[0].x[j+8] = exp2f(__half2float(qk_frag[i].x[j] - a_max.x));
                qk_exp_frag[0].x[j+1] = qk_exp_frag[0].x[j+9] = exp2f(__half2float(qk_frag[i].x[j+1] - a_max.x));
                qk_exp_frag[0].x[j+2] = qk_exp_frag[0].x[j+10] = exp2f(__half2float(qk_frag[i].x[j+2] - a_max.y));
                qk_exp_frag[0].x[j+3] = qk_exp_frag[0].x[j+11] = exp2f(__half2float(qk_frag[i].x[j+3] - a_max.y));
            }

            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qk_exp_frag[1].x[j] = qk_exp_frag[1].x[j+8] = exp2f(__half2float(qk_frag[i+10].x[j] - b_max.x));
                qk_exp_frag[1].x[j+1] = qk_exp_frag[1].x[j+9] = exp2f(__half2float(qk_frag[i+10].x[j+1] - b_max.x));
                qk_exp_frag[1].x[j+2] = qk_exp_frag[1].x[j+10] = exp2f(__half2float(qk_frag[i+10].x[j+2] - b_max.y));
                qk_exp_frag[1].x[j+3] = qk_exp_frag[1].x[j+11] = exp2f(__half2float(qk_frag[i+10].x[j+3] - b_max.y));
            }

            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qk_exp_frag[2].x[j] = qk_exp_frag[2].x[j+8] = exp2f(__half2float(qk_frag[i+20].x[j] - b_max.x));
                qk_exp_frag[2].x[j+1] = qk_exp_frag[2].x[j+9] = exp2f(__half2float(qk_frag[i+20].x[j+1] - b_max.x));
                qk_exp_frag[2].x[j+2] = qk_exp_frag[2].x[j+10] = exp2f(__half2float(qk_frag[i+20].x[j+2] - b_max.y));
                qk_exp_frag[2].x[j+3] = qk_exp_frag[2].x[j+11] = exp2f(__half2float(qk_frag[i+20].x[j+3] - b_max.y));
            }

            #pragma unroll
            for(int j = 0; j < 4; j++){
                wmma::load_matrix_sync(v_frag, v_sh + i*16*72 + j*16, 72);
                wmma::mma_sync(qkv_frag[j], qk_exp_frag[0], v_frag, qkv_frag[j]);
                wmma::mma_sync(qkv_frag[j+4], qk_exp_frag[1], v_frag, qkv_frag[j+4]);
                wmma::mma_sync(qkv_frag[j+8], qk_exp_frag[2], v_frag, qkv_frag[j+8]);
            }

            a_sum.x += qk_exp_frag[0].x[0] + qk_exp_frag[0].x[1];
            a_sum.y += qk_exp_frag[0].x[2] + qk_exp_frag[0].x[3];
            a_sum.x += qk_exp_frag[0].x[4] + qk_exp_frag[0].x[5];
            a_sum.y += qk_exp_frag[0].x[6] + qk_exp_frag[0].x[7];

            b_sum.x += qk_exp_frag[1].x[0] + qk_exp_frag[1].x[1];
            b_sum.y += qk_exp_frag[1].x[2] + qk_exp_frag[1].x[3];
            b_sum.x += qk_exp_frag[1].x[4] + qk_exp_frag[1].x[5];
            b_sum.y += qk_exp_frag[1].x[6] + qk_exp_frag[1].x[7];
            
            c_sum.x += qk_exp_frag[2].x[0] + qk_exp_frag[2].x[1];
            c_sum.y += qk_exp_frag[2].x[2] + qk_exp_frag[2].x[3];
            c_sum.x += qk_exp_frag[2].x[4] + qk_exp_frag[2].x[5];
            c_sum.y += qk_exp_frag[2].x[6] + qk_exp_frag[2].x[7];
        }        

        a_running_max = a_max;
        b_running_max = b_max;
        c_running_max = c_max;
    }

    #pragma unroll
    for(int j = 2; j > 0; j >>= 1){
        a_sum += __shfl_xor_sync(uint32_t(-1), a_sum, j);
        b_sum += __shfl_xor_sync(uint32_t(-1), b_sum, j);
        c_sum += __shfl_xor_sync(uint32_t(-1), c_sum, j);
    }

    #pragma unroll
    for(int i = 0; i < 4; i++){
        qkv_frag[i].x[0] /= a_sum.x;
        qkv_frag[i].x[1] /= a_sum.x;
        qkv_frag[i].x[2] /= a_sum.y;
        qkv_frag[i].x[3] /= a_sum.y;
        qkv_frag[i].x[4] /= a_sum.x;
        qkv_frag[i].x[5] /= a_sum.x;
        qkv_frag[i].x[6] /= a_sum.y;
        qkv_frag[i].x[7] /= a_sum.y;
        wmma::store_matrix_sync(O + i*16, qkv_frag[i], 64, wmma::mem_row_major);
    }
    O += 16*64;
    #pragma unroll
    for(int i = 4; i < 8; i++){
        qkv_frag[i].x[0] /= b_sum.x;
        qkv_frag[i].x[1] /= b_sum.x;
        qkv_frag[i].x[2] /= b_sum.y;
        qkv_frag[i].x[3] /= b_sum.y;
        qkv_frag[i].x[4] /= b_sum.x;
        qkv_frag[i].x[5] /= b_sum.x;
        qkv_frag[i].x[6] /= b_sum.y;
        qkv_frag[i].x[7] /= b_sum.y;
        wmma::store_matrix_sync(O + (i-4)*16, qkv_frag[i], 64, wmma::mem_row_major);
    }
    O += 16*64;
    #pragma unroll
    for(int i = 8; i < 12; i++){
        qkv_frag[i].x[0] /= c_sum.x;
        qkv_frag[i].x[1] /= c_sum.x;
        qkv_frag[i].x[2] /= c_sum.y;
        qkv_frag[i].x[3] /= c_sum.y;
        qkv_frag[i].x[4] /= c_sum.x;
        qkv_frag[i].x[5] /= c_sum.x;
        qkv_frag[i].x[6] /= c_sum.y;
        qkv_frag[i].x[7] /= c_sum.y;
        wmma::store_matrix_sync(O + (i-8)*16, qkv_frag[i], 64, wmma::mem_row_major);
    }
}


__global__ void flash10_async_q_inregs_64x64x160(half* Q, half* K, half* V, half* O, int s){
    
    Q += blockIdx.z * s * 64 + blockIdx.x * 256*64;
    O += blockIdx.z * s * 64 + blockIdx.x * 256*64;
    K += blockIdx.z * s * 64;
    V += blockIdx.z * s * 64;

    __shared__ half k_sh[160*72];
    __shared__ half v_sh[160*72];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag[16];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qk_exp_frag[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> qk_frag[40];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> qkv_frag[16];

    int wid = (threadIdx.x / 32);
    Q += 4*wid*16*64;
    O += 4*wid*16*64;

    const half scale = 1.f/sqrtf(64.f);
    
    #pragma unroll
    for(int i = 0; i < 16; i++){
        wmma::load_matrix_sync(q_frag[i], Q + (i/4)*16*64 + (i%4)*16, 64);
        wmma::fill_fragment(qkv_frag[i], 0.0f);
        
        #pragma unroll
        for(int j = 0; j < q_frag[i].num_elements; j++){
            q_frag[i].x[j] *= scale;
        }
    }

    half2 a_running_max = __float2half2_rn(0.f);
    half2 b_running_max = __float2half2_rn(0.f);
    half2 c_running_max = __float2half2_rn(0.f);
    half2 d_running_max = __float2half2_rn(0.f);
    
    half2 a_sum = __float2half2_rn(0.f);
    half2 b_sum = __float2half2_rn(0.f);
    half2 c_sum = __float2half2_rn(0.f);
    half2 d_sum = __float2half2_rn(0.f);


    for(int o = 0; o < s; o += 160){
        for(int i = threadIdx.x; i < 160 * 64 / 8; i += blockDim.x){
            int j = i + i/8;
            uint32_t smem = __cvta_generic_to_shared(reinterpret_cast<float4*>(k_sh)+j);
            CP_ASYNC_CG(smem, reinterpret_cast<float4*>(K) + i, 16);
            
            smem = __cvta_generic_to_shared(reinterpret_cast<float4*>(v_sh)+j);
            CP_ASYNC_CG(smem, reinterpret_cast<float4*>(V) + i, 16);
        }
        CP_ASYNC_WAIT_ALL();
        __syncthreads();
        
        K += 160*64; V += 160*64;
        half2 a_max = a_running_max;
        half2 b_max = b_running_max;
        half2 c_max = c_running_max;
        half2 d_max = d_running_max;
        
        #pragma unroll
        for(int i = 0; i < 10; i++){
            wmma::fill_fragment(qk_frag[i], 0.0f);
            wmma::fill_fragment(qk_frag[i+10], 0.0f);
            wmma::fill_fragment(qk_frag[i+20], 0.0f);
            wmma::fill_fragment(qk_frag[i+30], 0.0f);

            #pragma unroll
            for(int j = 0; j < 4; j++){
                wmma::load_matrix_sync(k_frag, k_sh + i*16*72 + j*16, 72);
                wmma::mma_sync(qk_frag[i], q_frag[j], k_frag, qk_frag[i]);
                wmma::mma_sync(qk_frag[i+10], q_frag[j+4], k_frag, qk_frag[i+10]);
                wmma::mma_sync(qk_frag[i+20], q_frag[j+8], k_frag, qk_frag[i+20]);
                wmma::mma_sync(qk_frag[i+30], q_frag[j+12], k_frag, qk_frag[i+30]);
            }
        }
        
        #pragma unroll
        for(int i = 0; i < 10; i++){
            a_max.x = max(a_max.x, qk_frag[i].x[0]);
            a_max.x = max(a_max.x, qk_frag[i].x[1]);
            a_max.y = max(a_max.y, qk_frag[i].x[2]);
            a_max.y = max(a_max.y, qk_frag[i].x[3]);
            a_max.x = max(a_max.x, qk_frag[i].x[4]);
            a_max.x = max(a_max.x, qk_frag[i].x[5]);
            a_max.y = max(a_max.y, qk_frag[i].x[6]);
            a_max.y = max(a_max.y, qk_frag[i].x[7]);

            b_max.x = max(b_max.x, qk_frag[i+10].x[0]);
            b_max.x = max(b_max.x, qk_frag[i+10].x[1]);
            b_max.y = max(b_max.y, qk_frag[i+10].x[2]);
            b_max.y = max(b_max.y, qk_frag[i+10].x[3]);
            b_max.x = max(b_max.x, qk_frag[i+10].x[4]);
            b_max.x = max(b_max.x, qk_frag[i+10].x[5]);
            b_max.y = max(b_max.y, qk_frag[i+10].x[6]);
            b_max.y = max(b_max.y, qk_frag[i+10].x[7]);

            c_max.x = max(c_max.x, qk_frag[i+20].x[0]);
            c_max.x = max(c_max.x, qk_frag[i+20].x[1]);
            c_max.y = max(c_max.y, qk_frag[i+20].x[2]);
            c_max.y = max(c_max.y, qk_frag[i+20].x[3]);
            c_max.x = max(c_max.x, qk_frag[i+20].x[4]);
            c_max.x = max(c_max.x, qk_frag[i+20].x[5]);
            c_max.y = max(c_max.y, qk_frag[i+20].x[6]);
            c_max.y = max(c_max.y, qk_frag[i+20].x[7]);

            d_max.x = max(d_max.x, qk_frag[i+30].x[0]);
            d_max.x = max(d_max.x, qk_frag[i+30].x[1]);
            d_max.y = max(d_max.y, qk_frag[i+30].x[2]);
            d_max.y = max(d_max.y, qk_frag[i+30].x[3]);
            d_max.x = max(d_max.x, qk_frag[i+30].x[4]);
            d_max.x = max(d_max.x, qk_frag[i+30].x[5]);
            d_max.y = max(d_max.y, qk_frag[i+30].x[6]);
            d_max.y = max(d_max.y, qk_frag[i+30].x[7]);
        }

        #pragma unroll
        for(int j = 2; j > 0; j >>= 1){
            a_max.x = max(a_max.x, __shfl_xor_sync(uint32_t(-1), a_max.x, j));
            a_max.y = max(a_max.y, __shfl_xor_sync(uint32_t(-1), a_max.y, j));
            b_max.x = max(b_max.x, __shfl_xor_sync(uint32_t(-1), b_max.x, j));
            b_max.y = max(b_max.y, __shfl_xor_sync(uint32_t(-1), b_max.y, j));
            c_max.x = max(c_max.x, __shfl_xor_sync(uint32_t(-1), c_max.x, j));
            c_max.y = max(c_max.y, __shfl_xor_sync(uint32_t(-1), c_max.y, j));
            d_max.x = max(d_max.x, __shfl_xor_sync(uint32_t(-1), d_max.x, j));
            d_max.y = max(d_max.y, __shfl_xor_sync(uint32_t(-1), d_max.y, j));
        }

        half rescale_x = exp2f(__half2float(a_running_max.x-a_max.x));
        half rescale_y = exp2f(__half2float(a_running_max.y-a_max.y));
        
        a_sum.x *= rescale_x;
        a_sum.y *= rescale_y;
        
        #pragma unroll
        for(int i = 0; i < 4; i ++){
            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qkv_frag[i].x[j] *= rescale_x;
                qkv_frag[i].x[j+1] *= rescale_x;
                qkv_frag[i].x[j+2] *= rescale_y;
                qkv_frag[i].x[j+3] *= rescale_y;
            }
        }
        rescale_x = exp2f(__half2float(b_running_max.x-b_max.x));
        rescale_y = exp2f(__half2float(b_running_max.y-b_max.y));
        
        b_sum.x *= rescale_x;
        b_sum.y *= rescale_y;
         
        #pragma unroll
        for(int i = 4; i < 8; i ++){
            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qkv_frag[i].x[j] *= rescale_x;
                qkv_frag[i].x[j+1] *= rescale_x;
                qkv_frag[i].x[j+2] *= rescale_y;
                qkv_frag[i].x[j+3] *= rescale_y;
            }
        }

        rescale_x = exp2f(__half2float(c_running_max.x-c_max.x));
        rescale_y = exp2f(__half2float(c_running_max.y-c_max.y));
        
        c_sum.x *= rescale_x;
        c_sum.y *= rescale_y;
         
        #pragma unroll
        for(int i = 8; i < 12; i ++){
            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qkv_frag[i].x[j] *= rescale_x;
                qkv_frag[i].x[j+1] *= rescale_x;
                qkv_frag[i].x[j+2] *= rescale_y;
                qkv_frag[i].x[j+3] *= rescale_y;
            }
        }

        rescale_x = exp2f(__half2float(d_running_max.x-d_max.x));
        rescale_y = exp2f(__half2float(d_running_max.y-d_max.y));
        
        d_sum.x *= rescale_x;
        d_sum.y *= rescale_y;
         
        #pragma unroll
        for(int i = 12; i < 16; i ++){
            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qkv_frag[i].x[j] *= rescale_x;
                qkv_frag[i].x[j+1] *= rescale_x;
                qkv_frag[i].x[j+2] *= rescale_y;
                qkv_frag[i].x[j+3] *= rescale_y;
            }
        }
        #pragma unroll
        for(int i = 0; i < 10; i++){
            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qk_exp_frag[0].x[j] = qk_exp_frag[0].x[j+8] = exp2f(__half2float(qk_frag[i].x[j] - a_max.x));
                qk_exp_frag[0].x[j+1] = qk_exp_frag[0].x[j+9] = exp2f(__half2float(qk_frag[i].x[j+1] - a_max.x));
                qk_exp_frag[0].x[j+2] = qk_exp_frag[0].x[j+10] = exp2f(__half2float(qk_frag[i].x[j+2] - a_max.y));
                qk_exp_frag[0].x[j+3] = qk_exp_frag[0].x[j+11] = exp2f(__half2float(qk_frag[i].x[j+3] - a_max.y));
            }

            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qk_exp_frag[1].x[j] = qk_exp_frag[1].x[j+8] = exp2f(__half2float(qk_frag[i+10].x[j] - b_max.x));
                qk_exp_frag[1].x[j+1] = qk_exp_frag[1].x[j+9] = exp2f(__half2float(qk_frag[i+10].x[j+1] - b_max.x));
                qk_exp_frag[1].x[j+2] = qk_exp_frag[1].x[j+10] = exp2f(__half2float(qk_frag[i+10].x[j+2] - b_max.y));
                qk_exp_frag[1].x[j+3] = qk_exp_frag[1].x[j+11] = exp2f(__half2float(qk_frag[i+10].x[j+3] - b_max.y));
            }

            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qk_exp_frag[2].x[j] = qk_exp_frag[2].x[j+8] = exp2f(__half2float(qk_frag[i+20].x[j] - b_max.x));
                qk_exp_frag[2].x[j+1] = qk_exp_frag[2].x[j+9] = exp2f(__half2float(qk_frag[i+20].x[j+1] - b_max.x));
                qk_exp_frag[2].x[j+2] = qk_exp_frag[2].x[j+10] = exp2f(__half2float(qk_frag[i+20].x[j+2] - b_max.y));
                qk_exp_frag[2].x[j+3] = qk_exp_frag[2].x[j+11] = exp2f(__half2float(qk_frag[i+20].x[j+3] - b_max.y));
            }

            #pragma unroll
            for(int j = 0; j < 5; j += 4){
                qk_exp_frag[3].x[j] = qk_exp_frag[3].x[j+8] = exp2f(__half2float(qk_frag[i+30].x[j] - b_max.x));
                qk_exp_frag[3].x[j+1] = qk_exp_frag[3].x[j+9] = exp2f(__half2float(qk_frag[i+30].x[j+1] - b_max.x));
                qk_exp_frag[3].x[j+2] = qk_exp_frag[3].x[j+10] = exp2f(__half2float(qk_frag[i+30].x[j+2] - b_max.y));
                qk_exp_frag[3].x[j+3] = qk_exp_frag[3].x[j+11] = exp2f(__half2float(qk_frag[i+30].x[j+3] - b_max.y));
            }

            #pragma unroll
            for(int j = 0; j < 4; j++){
                wmma::load_matrix_sync(v_frag, v_sh + i*16*72 + j*16, 72);
                wmma::mma_sync(qkv_frag[j], qk_exp_frag[0], v_frag, qkv_frag[j]);
                wmma::mma_sync(qkv_frag[j+4], qk_exp_frag[1], v_frag, qkv_frag[j+4]);
                wmma::mma_sync(qkv_frag[j+8], qk_exp_frag[2], v_frag, qkv_frag[j+8]);
                wmma::mma_sync(qkv_frag[j+12], qk_exp_frag[3], v_frag, qkv_frag[j+12]);
            }

            a_sum.x += qk_exp_frag[0].x[0] + qk_exp_frag[0].x[1];
            a_sum.y += qk_exp_frag[0].x[2] + qk_exp_frag[0].x[3];
            a_sum.x += qk_exp_frag[0].x[4] + qk_exp_frag[0].x[5];
            a_sum.y += qk_exp_frag[0].x[6] + qk_exp_frag[0].x[7];

            b_sum.x += qk_exp_frag[1].x[0] + qk_exp_frag[1].x[1];
            b_sum.y += qk_exp_frag[1].x[2] + qk_exp_frag[1].x[3];
            b_sum.x += qk_exp_frag[1].x[4] + qk_exp_frag[1].x[5];
            b_sum.y += qk_exp_frag[1].x[6] + qk_exp_frag[1].x[7];
            
            c_sum.x += qk_exp_frag[2].x[0] + qk_exp_frag[2].x[1];
            c_sum.y += qk_exp_frag[2].x[2] + qk_exp_frag[2].x[3];
            c_sum.x += qk_exp_frag[2].x[4] + qk_exp_frag[2].x[5];
            c_sum.y += qk_exp_frag[2].x[6] + qk_exp_frag[2].x[7];
            
            d_sum.x += qk_exp_frag[3].x[0] + qk_exp_frag[3].x[1];
            d_sum.y += qk_exp_frag[3].x[2] + qk_exp_frag[3].x[3];
            d_sum.x += qk_exp_frag[3].x[4] + qk_exp_frag[3].x[5];
            d_sum.y += qk_exp_frag[3].x[6] + qk_exp_frag[3].x[7];
        }    

        a_running_max = a_max;
        b_running_max = b_max;
        c_running_max = c_max;
        d_running_max = d_max;
    }

    #pragma unroll
    for(int j = 2; j > 0; j >>= 1){
        a_sum += __shfl_xor_sync(uint32_t(-1), a_sum, j);
        b_sum += __shfl_xor_sync(uint32_t(-1), b_sum, j);
        c_sum += __shfl_xor_sync(uint32_t(-1), c_sum, j);
        d_sum += __shfl_xor_sync(uint32_t(-1), d_sum, j);
    }

    #pragma unroll
    for(int i = 0; i < 4; i++){
        qkv_frag[i].x[0] /= a_sum.x;
        qkv_frag[i].x[1] /= a_sum.x;
        qkv_frag[i].x[2] /= a_sum.y;
        qkv_frag[i].x[3] /= a_sum.y;
        qkv_frag[i].x[4] /= a_sum.x;
        qkv_frag[i].x[5] /= a_sum.x;
        qkv_frag[i].x[6] /= a_sum.y;
        qkv_frag[i].x[7] /= a_sum.y;
        wmma::store_matrix_sync(O + i*16, qkv_frag[i], 64, wmma::mem_row_major);
    }
    O += 16*64;
    #pragma unroll
    for(int i = 4; i < 8; i++){
        qkv_frag[i].x[0] /= b_sum.x;
        qkv_frag[i].x[1] /= b_sum.x;
        qkv_frag[i].x[2] /= b_sum.y;
        qkv_frag[i].x[3] /= b_sum.y;
        qkv_frag[i].x[4] /= b_sum.x;
        qkv_frag[i].x[5] /= b_sum.x;
        qkv_frag[i].x[6] /= b_sum.y;
        qkv_frag[i].x[7] /= b_sum.y;
        wmma::store_matrix_sync(O + (i-4)*16, qkv_frag[i], 64, wmma::mem_row_major);
    }
    O += 16*64;
    #pragma unroll
    for(int i = 8; i < 12; i++){
        qkv_frag[i].x[0] /= c_sum.x;
        qkv_frag[i].x[1] /= c_sum.x;
        qkv_frag[i].x[2] /= c_sum.y;
        qkv_frag[i].x[3] /= c_sum.y;
        qkv_frag[i].x[4] /= c_sum.x;
        qkv_frag[i].x[5] /= c_sum.x;
        qkv_frag[i].x[6] /= c_sum.y;
        qkv_frag[i].x[7] /= c_sum.y;
        wmma::store_matrix_sync(O + (i-8)*16, qkv_frag[i], 64, wmma::mem_row_major);
    }

    O += 16*64;
    #pragma unroll
    for(int i = 12; i < 16; i++){
        qkv_frag[i].x[0] /= d_sum.x;
        qkv_frag[i].x[1] /= d_sum.x;
        qkv_frag[i].x[2] /= d_sum.y;
        qkv_frag[i].x[3] /= d_sum.y;
        qkv_frag[i].x[4] /= d_sum.x;
        qkv_frag[i].x[5] /= d_sum.x;
        qkv_frag[i].x[6] /= d_sum.y;
        qkv_frag[i].x[7] /= d_sum.y;
        wmma::store_matrix_sync(O + (i-12)*16, qkv_frag[i], 64, wmma::mem_row_major);
    }
}





int main(){
    // CHECK block size of 64, 128, 256, 384, 512
    // nvcc -Xptxas -O3,-v -maxrregcount=96 --expt-relaxed-constexpr -fmad=true -ftz=true flash4.cu -arch=sm_80 -o flash3  && ./flash3
    // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Dropout.cu

    int b = 12*16;
    int s = 1024*25;
    int d = 64;

    // int b = 1;
    // int s = 64;
    // int d = 64;

    half *Q = (half*)malloc(b*s*d*sizeof(half));
    half *K = (half*)malloc(b*s*d*sizeof(half));
    half *V = (half*)malloc(b*s*d*sizeof(half));
    half *O = (half*)malloc(b*s*d*sizeof(half));
    
    int ind[16*16];

    for(int i = 0; i < 16*16; i++){
        ind[i] = i/16;
    }

    ifstream is1("Query.txt");
    ifstream is2("Key.txt");
    ifstream is3("Value.txt");
    ifstream is4("Out.txt");
    ofstream os("Outflash.txt");

    for(int i = 0; i < b*s*d; i++){
        float a;
        is1>>a;
        Q[i] = __float2half(a);
        is2>>a;
        K[i] = __float2half(a);
        is3>>a;
        V[i] = __float2half(a);
        O[i] = 0.0f;
    }

    half *dev_Q, *dev_K, *dev_V;
    half *dev_O;
    int* dev_ind;

    cudaMalloc((void**)&dev_Q, b*s*d*sizeof(half));
    cudaMalloc((void**)&dev_K, b*s*d*sizeof(half));
    cudaMalloc((void**)&dev_V, b*s*d*sizeof(half));
    cudaMalloc((void**)&dev_O, b*s*d*sizeof(half));
    cudaMalloc((void**)&dev_ind, 16*16*sizeof(int));

    cudaMemcpy(dev_Q, Q, b*s*d*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_K, K, b*s*d*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V, V, b*s*d*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ind, ind, 16*16*sizeof(int),cudaMemcpyHostToDevice);
    
    dim3 blocks128(s/128, 1, b);
    dim3 blocks64(s/64, 1, b);
    dim3 blocks256(s/256, 1, b);
    dim3 blocks192((s+191)/192, 1, b);
    dim3 blocks96((s+95)/96, 1, b);
    
    flash10_async_q_inregs_16x64x160<<<blocks64, 128>>>(dev_Q, dev_K, dev_V, dev_O, s);
    
    flash10_async_q_inregs_32x64x160<<<blocks128, 128>>>(dev_Q, dev_K, dev_V, dev_O, s);

    flash10_async_q_inregs_48x64x160<<<blocks192, 128>>>(dev_Q, dev_K, dev_V, dev_O, s);

    flash10_async_q_inregs_64x64x160<<<blocks256, 128>>>(dev_Q, dev_K, dev_V, dev_O, s);
    
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("Error: %s", cudaGetErrorString(error));
    }

    cudaMemcpy(O, dev_O, b*s*d*sizeof(half), cudaMemcpyDeviceToHost);
    
    float max_diff = 0;
    float denom = 0;
    
    os<<std::fixed<<std::setprecision(6);
    std::cout<<endl<<endl<<endl;
    for(int i = 0; i < b*s*d; i++){
        if(__half2float(O[i]) < 0){
            os<<__half2float(O[i])<<" ";
        }else{
            os<<" "<<__half2float(O[i])<<" ";
        }
        
        if((i+1) % d == 0){
            os<<"\n";
        }
    }
    printf("\nmax denom: %f \n", denom);

    printf("max abs diff: %f \n", max_diff);

    free(Q);
    free(K);
    free(V);
    free(O);
    cudaFree(dev_ind);
    cudaFree(dev_Q);
    cudaFree(dev_K);
    cudaFree(dev_V);
    cudaFree(dev_O);
}