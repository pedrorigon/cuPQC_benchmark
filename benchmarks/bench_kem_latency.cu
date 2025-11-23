#include <vector>
#include <string>
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
#include <cupqc.hpp>

using namespace cupqc;

// KeyGen Kernel
template<typename KeyOp>
__global__ void keygen_kernel(uint8_t* pk, uint8_t* sk, uint8_t* ws, uint8_t* rnd) {
    __shared__ uint8_t smem[KeyOp::shared_memory_size];
    int i = blockIdx.x;
    KeyOp().execute(
        pk  + i * KeyOp::public_key_size,
        sk  + i * KeyOp::secret_key_size,
        rnd + i * KeyOp::entropy_size,
        ws  + i * KeyOp::workspace_size,
        smem
    );
}

// Encapsulation Kernel
template<typename EncOp>
__global__ void encaps_kernel(uint8_t* ct, uint8_t* ss, const uint8_t* pk, uint8_t* ws, uint8_t* rnd) {
    __shared__ uint8_t smem[EncOp::shared_memory_size];
    int i = blockIdx.x;
    EncOp().execute(
        ct  + i * EncOp::ciphertext_size,
        ss  + i * EncOp::shared_secret_size,
        pk  + i * EncOp::public_key_size,
        rnd + i * EncOp::entropy_size,
        ws  + i * EncOp::workspace_size,
        smem
    );
}

// Decapsulation Kernel
template<typename DecOp>
__global__ void decaps_kernel(uint8_t* ss, const uint8_t* ct, const uint8_t* sk, uint8_t* ws) {
    __shared__ uint8_t smem[DecOp::shared_memory_size];
    int i = blockIdx.x;
    DecOp().execute(
        ss  + i * DecOp::shared_secret_size,
        ct  + i * DecOp::ciphertext_size,
        sk  + i * DecOp::secret_key_size,
        ws  + i * DecOp::workspace_size,
        smem
    );
}

template<typename KeyOp, typename EncOp, typename DecOp>
struct LatencyBench {
    static void run(const char* tag) {
        const unsigned int single = 1;  // Single operation only
        
        std::vector<uint8_t> h_pk(single * KeyOp::public_key_size);
        std::vector<uint8_t> h_sk(single * KeyOp::secret_key_size);
        std::vector<uint8_t> h_ct(single * EncOp::ciphertext_size);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float ms;
        
        // KeyGen latency
        {
            auto ws_k = make_workspace<KeyOp>(single);
            auto rnd_k = get_entropy<KeyOp>(single);
            uint8_t *d_pk, *d_sk;
            cudaMalloc(&d_pk, single * KeyOp::public_key_size);
            cudaMalloc(&d_sk, single * KeyOp::secret_key_size);
            
            cudaEventRecord(start);
            keygen_kernel<KeyOp><<<single, KeyOp::BlockDim>>>(d_pk, d_sk, ws_k, rnd_k);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            
            printf("%s KeyGen: %.2f\n", tag, ms * 1000.0);  // Convert to microseconds
            
            cudaMemcpy(h_pk.data(), d_pk, h_pk.size(), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_sk.data(), d_sk, h_sk.size(), cudaMemcpyDeviceToHost);
            cudaFree(d_pk); cudaFree(d_sk);
            destroy_workspace(ws_k); release_entropy(rnd_k);
        }
        
        // Encaps latency
        {
            auto ws_e = make_workspace<EncOp>(single);
            auto rnd_e = get_entropy<EncOp>(single);
            uint8_t *d_ct, *d_ss1, *d_pk;
            cudaMalloc(&d_ct, single * EncOp::ciphertext_size);
            cudaMalloc(&d_ss1, single * EncOp::shared_secret_size);
            cudaMalloc(&d_pk, single * EncOp::public_key_size);
            cudaMemcpy(d_pk, h_pk.data(), h_pk.size(), cudaMemcpyHostToDevice);
            
            cudaEventRecord(start);
            encaps_kernel<EncOp><<<single, EncOp::BlockDim>>>(d_ct, d_ss1, d_pk, ws_e, rnd_e);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            
            printf("%s Encaps: %.2f\n", tag, ms * 1000.0);  // Convert to microseconds
            
            cudaMemcpy(h_ct.data(), d_ct, h_ct.size(), cudaMemcpyDeviceToHost);
            cudaFree(d_ct); cudaFree(d_ss1); cudaFree(d_pk);
            destroy_workspace(ws_e); release_entropy(rnd_e);
        }
        
        // Decaps latency
        {
            auto ws_d = make_workspace<DecOp>(single);
            uint8_t *d_ss2, *d_ct, *d_sk;
            cudaMalloc(&d_ss2, single * DecOp::shared_secret_size);
            cudaMalloc(&d_ct, single * DecOp::ciphertext_size);
            cudaMalloc(&d_sk, single * DecOp::secret_key_size);
            cudaMemcpy(d_ct, h_ct.data(), h_ct.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_sk, h_sk.data(), h_sk.size(), cudaMemcpyHostToDevice);
            
            cudaEventRecord(start);
            decaps_kernel<DecOp><<<single, DecOp::BlockDim>>>(d_ss2, d_ct, d_sk, ws_d);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            
            printf("%s Decaps: %.2f\n", tag, ms * 1000.0);  // Convert to microseconds
            
            cudaFree(d_ss2); cudaFree(d_ct); cudaFree(d_sk);
            destroy_workspace(ws_d);
        }
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

int main() {
    using K512Key = decltype( ML_KEM_512() + Function<function::Keygen>()   + Block() + BlockDim<128>() );
    using K512Enc = decltype( ML_KEM_512() + Function<function::Encaps>()   + Block() + BlockDim<128>() );
    using K512Dec = decltype( ML_KEM_512() + Function<function::Decaps>()   + Block() + BlockDim<128>() );

    using K768Key = decltype( ML_KEM_768() + Function<function::Keygen>()   + Block() + BlockDim<128>() );
    using K768Enc = decltype( ML_KEM_768() + Function<function::Encaps>()   + Block() + BlockDim<128>() );
    using K768Dec = decltype( ML_KEM_768() + Function<function::Decaps>()   + Block() + BlockDim<128>() );

    using K1024Key = decltype( ML_KEM_1024() + Function<function::Keygen>()   + Block() + BlockDim<128>() );
    using K1024Enc = decltype( ML_KEM_1024() + Function<function::Encaps>()   + Block() + BlockDim<128>() );
    using K1024Dec = decltype( ML_KEM_1024() + Function<function::Decaps>()   + Block() + BlockDim<128>() );

    // Measure single-operation latency
    LatencyBench<K512Key, K512Enc, K512Dec>::run("ML-KEM-512");
    LatencyBench<K768Key, K768Enc, K768Dec>::run("ML-KEM-768");
    LatencyBench<K1024Key, K1024Enc, K1024Dec>::run("ML-KEM-1024");

    printf("Latency measurements complete.\n");
    return 0;
}
