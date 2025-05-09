#include <vector>
#include <string>
#include <cstdio>
#include <cuda_runtime.h>
#include <cupqc.hpp>

using namespace cupqc;

// Generic kernels for keygen, sign, and verify
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

// Benchmark helper
void benchmark(const std::string& op_name, const cudaEvent_t& start, const cudaEvent_t& stop, unsigned int batch) {
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double s = ms / 1000.0;
    double thr = batch / s;
    printf("%s Throughput: %.2f ops/sec\n", op_name.c_str(), thr);
}

// Host wrappers for running benchmarks per variant
template <typename KeyOp>
void run_keygen(const std::string& variant, unsigned int batch,
                std::vector<uint8_t>& public_keys, std::vector<uint8_t>& secret_keys) {
    public_keys.resize(KeyOp::public_key_size * batch);
    secret_keys.resize(KeyOp::secret_key_size * batch);
    auto workspace   = make_workspace<KeyOp>(batch);
    auto randombytes = get_entropy<KeyOp>(batch);

    uint8_t *d_pk, *d_sk;
    cudaMalloc(&d_pk, public_keys.size());
    cudaMalloc(&d_sk, secret_keys.size());

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    keygen_kernel_generic<KeyOp><<<batch, KeyOp::BlockDim>>>(d_pk, d_sk, randombytes, workspace);
    cudaEventRecord(stop); cudaEventSynchronize(stop);

    cudaMemcpy(public_keys.data(), d_pk, public_keys.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(secret_keys.data(), d_sk, secret_keys.size(), cudaMemcpyDeviceToHost);
    benchmark(variant + " Key Generation", start, stop, batch);

    cudaFree(d_pk); cudaFree(d_sk);
    destroy_workspace(workspace);
    release_entropy(randombytes);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

template <typename SignOp>
void run_sign(const std::string& variant, unsigned int batch,
              std::vector<uint8_t>& signatures,
              const std::vector<uint8_t>& messages,
              const std::vector<uint8_t>& secret_keys,
              size_t message_size) {
    size_t sig_size = ((SignOp::signature_size + 7) / 8) * 8;
    signatures.resize(sig_size * batch);
    auto workspace   = make_workspace<SignOp>(batch);
    auto randombytes = get_entropy<SignOp>(batch);

    uint8_t *d_sig, *d_msg, *d_sk;
    cudaMalloc(&d_sig, signatures.size());
    cudaMalloc(&d_msg, messages.size());
    cudaMalloc(&d_sk, secret_keys.size());
    cudaMemcpy(d_msg, messages.data(), messages.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sk, secret_keys.data(), secret_keys.size(), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    sign_kernel_generic<SignOp><<<batch, SignOp::BlockDim>>>(d_sig, d_msg, message_size, d_sk, randombytes, workspace);
    cudaEventRecord(stop); cudaEventSynchronize(stop);

    cudaMemcpy(signatures.data(), d_sig, signatures.size(), cudaMemcpyDeviceToHost);
    benchmark(variant + " Signing", start, stop, batch);

    cudaFree(d_sig); cudaFree(d_msg); cudaFree(d_sk);
    destroy_workspace(workspace);
    release_entropy(randombytes);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <typename VerifyOp>
void run_verify(const std::string& variant, unsigned int batch,
                std::vector<uint8_t>& valids,
                const std::vector<uint8_t>& signatures,
                const std::vector<uint8_t>& messages,
                const std::vector<uint8_t>& public_keys,
                size_t message_size) {
    valids.resize(batch);
    auto workspace = make_workspace<VerifyOp>(batch);
    size_t sig_size = ((VerifyOp::signature_size + 7) / 8) * 8;

    uint8_t *d_sig, *d_msg, *d_pk, *d_valid;
    cudaMalloc(&d_sig,       sig_size * batch);
    cudaMalloc(&d_msg,       messages.size());
    cudaMalloc(&d_pk,        public_keys.size());
    cudaMalloc(&d_valid,     batch * sizeof(uint8_t));

    cudaMemcpy(d_sig, signatures.data(), sig_size * batch, cudaMemcpyHostToDevice);
    cudaMemcpy(d_msg, messages.data(),    messages.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pk,  public_keys.data(), public_keys.size(), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    verify_kernel_generic<VerifyOp><<<batch, VerifyOp::BlockDim>>>(d_valid, d_sig, d_msg, message_size, d_pk, workspace);
    cudaEventRecord(stop); cudaEventSynchronize(stop);

    cudaMemcpy(valids.data(), d_valid, batch * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    benchmark(variant + " Verification", start, stop, batch);

    cudaFree(d_sig); cudaFree(d_msg); cudaFree(d_pk); cudaFree(d_valid);
    destroy_workspace(workspace);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    unsigned int batch = 100000;
    const size_t message_size = 1024;

    // Prepare a dummy message buffer
    std::vector<uint8_t> messages(message_size * batch, 0xAB);

    // ML-DSA-44
    using MLDSA44KeyOp    = decltype(ML_DSA_44() + Function<function::Keygen>() + Block() + BlockDim<128>());
    using MLDSA44SignOp   = decltype(ML_DSA_44() + Function<function::Sign>()   + Block() + BlockDim<128>());
    using MLDSA44VerifyOp = decltype(ML_DSA_44() + Function<function::Verify>() + Block() + BlockDim<128>());

    std::vector<uint8_t> public_keys44, secret_keys44, signatures44, valids44;
    printf("\n=== ML-DSA-44 Benchmark ===\n");
    run_keygen<MLDSA44KeyOp>("ML-DSA-44", batch, public_keys44, secret_keys44);
    run_sign<MLDSA44SignOp>("ML-DSA-44", batch, signatures44, messages, secret_keys44, message_size);
    run_verify<MLDSA44VerifyOp>("ML-DSA-44", batch, valids44, signatures44, messages, public_keys44, message_size);

    // ML-DSA-65
    using MLDSA65KeyOp    = decltype(ML_DSA_65() + Function<function::Keygen>() + Block() + BlockDim<128>());
    using MLDSA65SignOp   = decltype(ML_DSA_65() + Function<function::Sign>()   + Block() + BlockDim<128>());
    using MLDSA65VerifyOp = decltype(ML_DSA_65() + Function<function::Verify>() + Block() + BlockDim<128>());

    std::vector<uint8_t> public_keys65, secret_keys65, signatures65, valids65;
    printf("\n=== ML-DSA-65 Benchmark ===\n");
    run_keygen<MLDSA65KeyOp>("ML-DSA-65", batch, public_keys65, secret_keys65);
    run_sign<MLDSA65SignOp>("ML-DSA-65", batch, signatures65, messages, secret_keys65, message_size);
    run_verify<MLDSA65VerifyOp>("ML-DSA-65", batch, valids65, signatures65, messages, public_keys65, message_size);

    // ML-DSA-87
    using MLDSA87KeyOp    = decltype(ML_DSA_87() + Function<function::Keygen>() + Block() + BlockDim<128>());
    using MLDSA87SignOp   = decltype(ML_DSA_87() + Function<function::Sign>()   + Block() + BlockDim<128>());
    using MLDSA87VerifyOp = decltype(ML_DSA_87() + Function<function::Verify>() + Block() + BlockDim<128>());

    std::vector<uint8_t> public_keys87, secret_keys87, signatures87, valids87;
    printf("\n=== ML-DSA-87 Benchmark ===\n");
    run_keygen<MLDSA87KeyOp>("ML-DSA-87", batch, public_keys87, secret_keys87);
    run_sign<MLDSA87SignOp>("ML-DSA-87", batch, signatures87, messages, secret_keys87, message_size);
    run_verify<MLDSA87VerifyOp>("ML-DSA-87", batch, valids87, signatures87, messages, public_keys87, message_size);

    printf("\nAll benchmarks completed successfully.\n");
    return 0;
}