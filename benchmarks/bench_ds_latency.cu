#include <vector>
#include <string>
#include <cstdio>
#include <cuda_runtime.h>
#include <cupqc.hpp>

using namespace cupqc;

template <typename KeyOp>
__global__ void keygen_kernel_generic(uint8_t* public_keys, uint8_t* secret_keys, uint8_t* randombytes, uint8_t* workspace) {
    __shared__ uint8_t smem_ptr[KeyOp::shared_memory_size];
    int idx = blockIdx.x;
    auto pk    = public_keys + idx * KeyOp::public_key_size;
    auto sk    = secret_keys + idx * KeyOp::secret_key_size;
    auto rnd   = randombytes + idx * KeyOp::entropy_size;
    auto work  = workspace + idx * KeyOp::workspace_size;
    KeyOp().execute(pk, sk, rnd, work, smem_ptr);
}

template <typename SignOp>
__global__ void sign_kernel_generic(uint8_t* signatures, const uint8_t* messages, size_t message_size,
                                   const uint8_t* secret_keys, uint8_t* randombytes, uint8_t* workspace) {
    __shared__ uint8_t smem_ptr[SignOp::shared_memory_size];
    int idx = blockIdx.x;
    auto sig  = signatures + idx * (((SignOp::signature_size + 7) / 8) * 8);
    auto msg  = messages   + idx * message_size;
    auto sk   = secret_keys + idx * SignOp::secret_key_size;
    auto rnd  = randombytes + idx * SignOp::entropy_size;
    auto work = workspace  + idx * SignOp::workspace_size;
    SignOp().execute(sig, msg, message_size, sk, rnd, work, smem_ptr);
}

template <typename VerifyOp>
__global__ void verify_kernel_generic(uint8_t* valids, const uint8_t* signatures, const uint8_t* messages, size_t message_size,
                                     const uint8_t* public_keys, uint8_t* workspace) {
    __shared__ uint8_t smem_ptr[VerifyOp::shared_memory_size];
    int idx = blockIdx.x;
    auto sig = signatures    + idx * (((VerifyOp::signature_size + 7) / 8) * 8);
    auto msg = messages      + idx * message_size;
    auto pk  = public_keys   + idx * VerifyOp::public_key_size;
    auto work = workspace    + idx * VerifyOp::workspace_size;
    valids[idx] = VerifyOp().execute(msg, message_size, sig, pk, work, smem_ptr) ? 1 : 0;
}

template <typename KeyOp, typename SignOp, typename VerifyOp>
struct DSALatencyBench {
    static void run(const char* tag) {
        const unsigned int single = 1;
        const size_t message_size = 1024;
        std::vector<uint8_t> messages(message_size * single, 0xAB);
        std::vector<uint8_t> public_keys, secret_keys, signatures;
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        float ms;
        
        // KeyGen latency
        {
            public_keys.resize(KeyOp::public_key_size * single);
            secret_keys.resize(KeyOp::secret_key_size * single);
            
            auto workspace = make_workspace<KeyOp>(single);
            auto randombytes = get_entropy<KeyOp>(single);
            uint8_t *d_pk, *d_sk;
            cudaMalloc(&d_pk, public_keys.size());
            cudaMalloc(&d_sk, secret_keys.size());
            
            cudaEventRecord(start);
            keygen_kernel_generic<KeyOp><<<single, KeyOp::BlockDim>>>(d_pk, d_sk, randombytes, workspace);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            
            printf("%s Key Generation: %.2f\n", tag, ms * 1000.0);  // Convert to microseconds
            
            cudaMemcpy(public_keys.data(), d_pk, public_keys.size(), cudaMemcpyDeviceToHost);
            cudaMemcpy(secret_keys.data(), d_sk, secret_keys.size(), cudaMemcpyDeviceToHost);
            cudaFree(d_pk); cudaFree(d_sk);
            destroy_workspace(workspace);
            release_entropy(randombytes);
        }
        
        // Signing latency
        {
            size_t sig_size = ((SignOp::signature_size + 7) / 8) * 8;
            signatures.resize(sig_size * single);
            
            auto workspace = make_workspace<SignOp>(single);
            auto randombytes = get_entropy<SignOp>(single);
            uint8_t *d_sig, *d_msg, *d_sk;
            cudaMalloc(&d_sig, signatures.size());
            cudaMalloc(&d_msg, messages.size());
            cudaMalloc(&d_sk, secret_keys.size());
            cudaMemcpy(d_msg, messages.data(), messages.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_sk, secret_keys.data(), secret_keys.size(), cudaMemcpyHostToDevice);
            
            cudaEventRecord(start);
            sign_kernel_generic<SignOp><<<single, SignOp::BlockDim>>>(d_sig, d_msg, message_size, d_sk, randombytes, workspace);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            
            printf("%s Signing: %.2f\n", tag, ms * 1000.0);  // Convert to microseconds
            
            cudaMemcpy(signatures.data(), d_sig, signatures.size(), cudaMemcpyDeviceToHost);
            cudaFree(d_sig); cudaFree(d_msg); cudaFree(d_sk);
            destroy_workspace(workspace);
            release_entropy(randombytes);
        }
        
        // Verification latency
        {
            size_t sig_size = ((VerifyOp::signature_size + 7) / 8) * 8;
            std::vector<uint8_t> valids(single);
            
            auto workspace = make_workspace<VerifyOp>(single);
            uint8_t *d_sig, *d_msg, *d_pk, *d_valid;
            cudaMalloc(&d_sig, sig_size * single);
            cudaMalloc(&d_msg, messages.size());
            cudaMalloc(&d_pk, public_keys.size());
            cudaMalloc(&d_valid, single * sizeof(uint8_t));
            cudaMemcpy(d_sig, signatures.data(), sig_size * single, cudaMemcpyHostToDevice);
            cudaMemcpy(d_msg, messages.data(), messages.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_pk, public_keys.data(), public_keys.size(), cudaMemcpyHostToDevice);
            
            cudaEventRecord(start);
            verify_kernel_generic<VerifyOp><<<single, VerifyOp::BlockDim>>>(d_valid, d_sig, d_msg, message_size, d_pk, workspace);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            
            printf("%s Verification: %.2f\n", tag, ms * 1000.0);  // Convert to microseconds
            
            cudaFree(d_sig); cudaFree(d_msg); cudaFree(d_pk); cudaFree(d_valid);
            destroy_workspace(workspace);
        }
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

int main() {
    using MLDSA44KeyOp = decltype(ML_DSA_44() + Function<function::Keygen>() + Block() + BlockDim<128>());
    using MLDSA44SignOp = decltype(ML_DSA_44() + Function<function::Sign>() + Block() + BlockDim<128>());
    using MLDSA44VerifyOp = decltype(ML_DSA_44() + Function<function::Verify>() + Block() + BlockDim<128>());
    
    using MLDSA65KeyOp = decltype(ML_DSA_65() + Function<function::Keygen>() + Block() + BlockDim<128>());
    using MLDSA65SignOp = decltype(ML_DSA_65() + Function<function::Sign>() + Block() + BlockDim<128>());
    using MLDSA65VerifyOp = decltype(ML_DSA_65() + Function<function::Verify>() + Block() + BlockDim<128>());
    
    using MLDSA87KeyOp = decltype(ML_DSA_87() + Function<function::Keygen>() + Block() + BlockDim<128>());
    using MLDSA87SignOp = decltype(ML_DSA_87() + Function<function::Sign>() + Block() + BlockDim<128>());
    using MLDSA87VerifyOp = decltype(ML_DSA_87() + Function<function::Verify>() + Block() + BlockDim<128>());
    
    DSALatencyBench<MLDSA44KeyOp, MLDSA44SignOp, MLDSA44VerifyOp>::run("ML-DSA-44");
    DSALatencyBench<MLDSA65KeyOp, MLDSA65SignOp, MLDSA65VerifyOp>::run("ML-DSA-65");
    DSALatencyBench<MLDSA87KeyOp, MLDSA87SignOp, MLDSA87VerifyOp>::run("ML-DSA-87");
    
    printf("Latency measurements complete.\n");
    return 0;
}
