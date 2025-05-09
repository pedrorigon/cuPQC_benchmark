#include <vector>
#include <string>
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
#include <cupqc.hpp>

using namespace cupqc;

template<typename T>
void benchmark(const std::string& tag,
               const std::string& op_name,
               const cudaEvent_t& start,
               const cudaEvent_t& stop,
               unsigned int batch) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double sec = ms / 1000.0;
    double thr = batch / sec;
    printf("[%s] %s: %.2f ops/sec\n", tag.c_str(), op_name.c_str(), thr);
}

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
struct KEMBench {
    static void run(const char* tag, unsigned int batch) {
        // Buffers host
        std::vector<uint8_t> h_pk(batch * KeyOp::public_key_size);
        std::vector<uint8_t> h_sk(batch * KeyOp::secret_key_size);
        std::vector<uint8_t> h_ct(batch * EncOp::ciphertext_size);
        std::vector<uint8_t> h_ss1(batch * EncOp::shared_secret_size);
        std::vector<uint8_t> h_ss2(batch * DecOp::shared_secret_size);

        auto ws_k = make_workspace<KeyOp>(batch);
        auto rnd_k = get_entropy<KeyOp>(batch);
        auto ws_e = make_workspace<EncOp>(batch);
        auto rnd_e = get_entropy<EncOp>(batch);
        auto ws_d = make_workspace<DecOp>(batch);

        // Buffers device
        uint8_t *d_pk, *d_sk, *d_ct, *d_ss1, *d_ss2;
        cudaMalloc(&d_pk, batch * KeyOp::public_key_size);
        cudaMalloc(&d_sk, batch * KeyOp::secret_key_size);
        cudaMalloc(&d_ct, batch * EncOp::ciphertext_size);
        cudaMalloc(&d_ss1, batch * EncOp::shared_secret_size);
        cudaMalloc(&d_ss2, batch * DecOp::shared_secret_size);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // --- Key Generation ---
        cudaEventRecord(start);
        keygen_kernel<KeyOp><<<batch, KeyOp::BlockDim>>>(d_pk, d_sk, ws_k, rnd_k);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaMemcpy(h_pk.data(), d_pk, h_pk.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sk.data(), d_sk, h_sk.size(), cudaMemcpyDeviceToHost);
        benchmark<KeyOp>(tag, "KeyGen", start, stop, batch);

        // --- Encapsulation ---
        cudaMemcpy(d_pk, h_pk.data(), h_pk.size(), cudaMemcpyHostToDevice);
        cudaEventRecord(start);
        encaps_kernel<EncOp><<<batch, EncOp::BlockDim>>>(d_ct, d_ss1, d_pk, ws_e, rnd_e);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaMemcpy(h_ct.data(), d_ct, h_ct.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ss1.data(), d_ss1, h_ss1.size(), cudaMemcpyDeviceToHost);
        benchmark<EncOp>(tag, "Encaps", start, stop, batch);

        // --- Decapsulation ---
        cudaMemcpy(d_ct, h_ct.data(), h_ct.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sk, h_sk.data(), h_sk.size(), cudaMemcpyHostToDevice);
        cudaEventRecord(start);
        decaps_kernel<DecOp><<<batch, DecOp::BlockDim>>>(d_ss2, d_ct, d_sk, ws_d);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaMemcpy(h_ss2.data(), d_ss2, h_ss2.size(), cudaMemcpyDeviceToHost);
        benchmark<DecOp>(tag, "Decaps", start, stop, batch);

        // Cleanup
        cudaFree(d_pk); cudaFree(d_sk);
        cudaFree(d_ct); cudaFree(d_ss1); cudaFree(d_ss2);
        destroy_workspace(ws_k); release_entropy(rnd_k);
        destroy_workspace(ws_e); release_entropy(rnd_e);
        destroy_workspace(ws_d);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
};

int main() {
    const unsigned int batch = 100000;

    using K512Key = decltype( ML_KEM_512() + Function<function::Keygen>()   + Block() + BlockDim<128>() );
    using K512Enc = decltype( ML_KEM_512() + Function<function::Encaps>()   + Block() + BlockDim<128>() );
    using K512Dec = decltype( ML_KEM_512() + Function<function::Decaps>()   + Block() + BlockDim<128>() );

    using K768Key = decltype( ML_KEM_768() + Function<function::Keygen>()   + Block() + BlockDim<128>() );
    using K768Enc = decltype( ML_KEM_768() + Function<function::Encaps>()   + Block() + BlockDim<128>() );
    using K768Dec = decltype( ML_KEM_768() + Function<function::Decaps>()   + Block() + BlockDim<128>() );

    using K1024Key = decltype( ML_KEM_1024() + Function<function::Keygen>()   + Block() + BlockDim<128>() );
    using K1024Enc = decltype( ML_KEM_1024() + Function<function::Encaps>()   + Block() + BlockDim<128>() );
    using K1024Dec = decltype( ML_KEM_1024() + Function<function::Decaps>()   + Block() + BlockDim<128>() );

    // Execute benchmarks
    KEMBench<K512Key, K512Enc, K512Dec>::run("ML-KEM-512", batch);
    KEMBench<K768Key, K768Enc, K768Dec>::run("ML-KEM-768", batch);
    KEMBench<K1024Key, K1024Enc, K1024Dec>::run("ML-KEM-1024", batch);

    printf("All benchmarks Finalized.\n");
    return 0;
}
