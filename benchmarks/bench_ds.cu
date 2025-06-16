#include <vector>
#include <string>
#include <cstdio>
#include <cuda_runtime.h>
#include <cupqc.hpp>
#include <nvml.h>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>

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

void benchmark(const std::string& op_name, const cudaEvent_t& start, const cudaEvent_t& stop, unsigned int batch,
               size_t mem_used, double peak_gpu_util) {
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double s = ms / 1000.0;
    double thr = batch / s;
    printf("%s\n", op_name.c_str());
    printf("  Throughput: %.2f ops/sec\n", thr);
    printf("  GPU Memory Used: %.2f MB\n", static_cast<double>(mem_used) / (1024 * 1024));
    printf("  Peak GPU Utilization: %.2f%%\n", peak_gpu_util);
}

void sample_gpu_utilization(nvmlDevice_t device, std::vector<unsigned int>& util_samples) {
    nvmlUtilization_t utilization;
    if (nvmlDeviceGetUtilizationRates(device, &utilization) == NVML_SUCCESS) {
        util_samples.push_back(utilization.gpu);
    }
}

template <typename KeyOp>
void run_keygen(nvmlDevice_t device, const std::string& variant, unsigned int batch,
                std::vector<uint8_t>& public_keys, std::vector<uint8_t>& secret_keys) {
    public_keys.resize(KeyOp::public_key_size * batch);
    secret_keys.resize(KeyOp::secret_key_size * batch);

    nvmlMemory_t mem_info_before;
    nvmlDeviceGetMemoryInfo(device, &mem_info_before);

    auto workspace = make_workspace<KeyOp>(batch);
    auto randombytes = get_entropy<KeyOp>(batch);

    uint8_t *d_pk, *d_sk;
    cudaMalloc(&d_pk, public_keys.size());
    cudaMalloc(&d_sk, secret_keys.size());

    nvmlMemory_t mem_info_after_alloc;
    nvmlDeviceGetMemoryInfo(device, &mem_info_after_alloc);
    size_t mem_used = mem_info_after_alloc.used - mem_info_before.used;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    std::vector<unsigned int> util_samples;
    cudaEventRecord(start);
    keygen_kernel_generic<KeyOp><<<batch, KeyOp::BlockDim>>>(d_pk, d_sk, randombytes, workspace);
    cudaEventRecord(stop);
    
    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        sample_gpu_utilization(device, util_samples);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    cudaEventSynchronize(stop);

    double peak_util = 0.0;
    if (!util_samples.empty()) {
        peak_util = *std::max_element(util_samples.begin(), util_samples.end());
    }

    cudaMemcpy(public_keys.data(), d_pk, public_keys.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(secret_keys.data(), d_sk, secret_keys.size(), cudaMemcpyDeviceToHost);
    benchmark(variant + " Key Generation", start, stop, batch, mem_used, peak_util);

    cudaFree(d_pk); cudaFree(d_sk);
    destroy_workspace(workspace);
    release_entropy(randombytes);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

template <typename SignOp>
void run_sign(nvmlDevice_t device, const std::string& variant, unsigned int batch,
              std::vector<uint8_t>& signatures,
              const std::vector<uint8_t>& messages,
              const std::vector<uint8_t>& secret_keys,
              size_t message_size) {
    size_t sig_size = ((SignOp::signature_size + 7) / 8) * 8;
    signatures.resize(sig_size * batch);

    nvmlMemory_t mem_info_before;
    nvmlDeviceGetMemoryInfo(device, &mem_info_before);

    auto workspace = make_workspace<SignOp>(batch);
    auto randombytes = get_entropy<SignOp>(batch);

    uint8_t *d_sig, *d_msg, *d_sk;
    cudaMalloc(&d_sig, signatures.size());
    cudaMalloc(&d_msg, messages.size());
    cudaMalloc(&d_sk, secret_keys.size());
    cudaMemcpy(d_msg, messages.data(), messages.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sk, secret_keys.data(), secret_keys.size(), cudaMemcpyHostToDevice);

    nvmlMemory_t mem_info_after_alloc;
    nvmlDeviceGetMemoryInfo(device, &mem_info_after_alloc);
    size_t mem_used = mem_info_after_alloc.used - mem_info_before.used;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    std::vector<unsigned int> util_samples;
    cudaEventRecord(start);
    sign_kernel_generic<SignOp><<<batch, SignOp::BlockDim>>>(d_sig, d_msg, message_size, d_sk, randombytes, workspace);
    cudaEventRecord(stop);
    
    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        sample_gpu_utilization(device, util_samples);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    cudaEventSynchronize(stop);

    double peak_util = 0.0;
    if (!util_samples.empty()) {
        peak_util = *std::max_element(util_samples.begin(), util_samples.end());
    }

    cudaMemcpy(signatures.data(), d_sig, signatures.size(), cudaMemcpyDeviceToHost);
    benchmark(variant + " Signing", start, stop, batch, mem_used, peak_util);

    cudaFree(d_sig); cudaFree(d_msg); cudaFree(d_sk);
    destroy_workspace(workspace);
    release_entropy(randombytes);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <typename VerifyOp>
void run_verify(nvmlDevice_t device, const std::string& variant, unsigned int batch,
                std::vector<uint8_t>& valids,
                const std::vector<uint8_t>& signatures,
                const std::vector<uint8_t>& messages,
                const std::vector<uint8_t>& public_keys,
                size_t message_size) {
    valids.resize(batch);

    nvmlMemory_t mem_info_before;
    nvmlDeviceGetMemoryInfo(device, &mem_info_before);

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

    nvmlMemory_t mem_info_after_alloc;
    nvmlDeviceGetMemoryInfo(device, &mem_info_after_alloc);
    size_t mem_used = mem_info_after_alloc.used - mem_info_before.used;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    std::vector<unsigned int> util_samples;
    cudaEventRecord(start);
    verify_kernel_generic<VerifyOp><<<batch, VerifyOp::BlockDim>>>(d_valid, d_sig, d_msg, message_size, d_pk, workspace);
    cudaEventRecord(stop);
    
    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        sample_gpu_utilization(device, util_samples);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    cudaEventSynchronize(stop);

    double peak_util = 0.0;
    if (!util_samples.empty()) {
        peak_util = *std::max_element(util_samples.begin(), util_samples.end());
    }

    cudaMemcpy(valids.data(), d_valid, batch * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    benchmark(variant + " Verification", start, stop, batch, mem_used, peak_util);

    cudaFree(d_sig); cudaFree(d_msg); cudaFree(d_pk); cudaFree(d_valid);
    destroy_workspace(workspace);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    nvmlReturn_t result = nvmlInit();
    if (NVML_SUCCESS != result) {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        return 1;
    }

    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (NVML_SUCCESS != result) {
        printf("Failed to get NVML device handle: %s\n", nvmlErrorString(result));
        nvmlShutdown();
        return 1;
    }

    unsigned int batch = 100000;
    const size_t message_size = 1024;
    std::vector<uint8_t> messages(message_size * batch, 0xAB);

    using MLDSA44KeyOp    = decltype(ML_DSA_44() + Function<function::Keygen>() + Block() + BlockDim<128>());
    using MLDSA44SignOp   = decltype(ML_DSA_44() + Function<function::Sign>()   + Block() + BlockDim<128>());
    using MLDSA44VerifyOp = decltype(ML_DSA_44() + Function<function::Verify>() + Block() + BlockDim<128>());

    std::vector<uint8_t> public_keys44, secret_keys44, signatures44, valids44;
    printf("\n=== ML-DSA-44 Benchmark ===\n");
    run_keygen<MLDSA44KeyOp>(device, "ML-DSA-44", batch, public_keys44, secret_keys44);
    run_sign<MLDSA44SignOp>(device, "ML-DSA-44", batch, signatures44, messages, secret_keys44, message_size);
    run_verify<MLDSA44VerifyOp>(device, "ML-DSA-44", batch, valids44, signatures44, messages, public_keys44, message_size);

    using MLDSA65KeyOp    = decltype(ML_DSA_65() + Function<function::Keygen>() + Block() + BlockDim<128>());
    using MLDSA65SignOp   = decltype(ML_DSA_65() + Function<function::Sign>()   + Block() + BlockDim<128>());
    using MLDSA65VerifyOp = decltype(ML_DSA_65() + Function<function::Verify>() + Block() + BlockDim<128>());

    std::vector<uint8_t> public_keys65, secret_keys65, signatures65, valids65;
    printf("\n=== ML-DSA-65 Benchmark ===\n");
    run_keygen<MLDSA65KeyOp>(device, "ML-DSA-65", batch, public_keys65, secret_keys65);
    run_sign<MLDSA65SignOp>(device, "ML-DSA-65", batch, signatures65, messages, secret_keys65, message_size);
    run_verify<MLDSA65VerifyOp>(device, "ML-DSA-65", batch, valids65, signatures65, messages, public_keys65, message_size);

    using MLDSA87KeyOp    = decltype(ML_DSA_87() + Function<function::Keygen>() + Block() + BlockDim<128>());
    using MLDSA87SignOp   = decltype(ML_DSA_87() + Function<function::Sign>()   + Block() + BlockDim<128>());
    using MLDSA87VerifyOp = decltype(ML_DSA_87() + Function<function::Verify>() + Block() + BlockDim<128>());

    std::vector<uint8_t> public_keys87, secret_keys87, signatures87, valids87;
    printf("\n=== ML-DSA-87 Benchmark ===\n");
    run_keygen<MLDSA87KeyOp>(device, "ML-DSA-87", batch, public_keys87, secret_keys87);
    run_sign<MLDSA87SignOp>(device, "ML-DSA-87", batch, signatures87, messages, secret_keys87, message_size);
    run_verify<MLDSA87VerifyOp>(device, "ML-DSA-87", batch, valids87, signatures87, messages, public_keys87, message_size);

    nvmlShutdown();
    printf("\nAll benchmarks completed successfully.\n");
    return 0;
}