#pragma once

#include <cuda_runtime.h>
#include <nvml.h>
#include <vector>
#include <string>
#include <cstdio>
#include <thread>
#include <chrono>
#include <algorithm>
#include <atomic>

class GpuProfiler {
public:
    GpuProfiler() {
        nvmlReturn_t result = nvmlInit();
        if (NVML_SUCCESS != result) {
            fprintf(stderr, "Failed to initialize NVML: %s\n", nvmlErrorString(result));
            initialized_ = false;
        } else {
            result = nvmlDeviceGetHandleByIndex(0, &device_);
            if (NVML_SUCCESS != result) {
                fprintf(stderr, "Failed to get NVML device handle: %s\n", nvmlErrorString(result));
                initialized_ = false;
            } else {
                initialized_ = true;
            }
        }
    }

    ~GpuProfiler() {
        if (initialized_) {
            nvmlShutdown();
        }
    }

    bool is_initialized() const { return initialized_; }

    void start_profile() {
        if (!initialized_) return;
        
        nvmlMemory_t mem_info;
        nvmlDeviceGetMemoryInfo(device_, &mem_info);
        start_memory_ = mem_info.used;
        peak_memory_ = mem_info.used;

        stop_sampling_ = false;
        util_samples_.clear();
        sampling_thread_ = std::thread([this]() {
            while (!stop_sampling_) {
                nvmlUtilization_t utilization;
                if (nvmlDeviceGetUtilizationRates(device_, &utilization) == NVML_SUCCESS) {
                    util_samples_.push_back(utilization.gpu);
                }
                
                nvmlMemory_t mem_info_sample;
                if (nvmlDeviceGetMemoryInfo(device_, &mem_info_sample) == NVML_SUCCESS) {
                    if (mem_info_sample.used > peak_memory_) {
                        peak_memory_ = mem_info_sample.used;
                    }
                }

                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });
    }

    void stop_profile(const std::string& op_name, const cudaEvent_t& start, const cudaEvent_t& stop, unsigned int batch) {
        if (!initialized_) return;

        stop_sampling_ = true;
        if (sampling_thread_.joinable()) {
            sampling_thread_.join();
        }

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        double s = ms / 1000.0;
        double thr = batch / s;

        double peak_util = 0.0;
        if (!util_samples_.empty()) {
            peak_util = *std::max_element(util_samples_.begin(), util_samples_.end());
        }

        printf("%s\n", op_name.c_str());
        printf("  Throughput: %.2f ops/sec\n", thr);
        printf("  Peak GPU Memory Used: %.2f MB\n", static_cast<double>(peak_memory_) / (1024 * 1024));
        printf("  Peak GPU Utilization: %.2f%%\n", peak_util);
    }

    nvmlDevice_t get_device() const { return device_; }

private:
    bool initialized_;
    nvmlDevice_t device_;
    size_t start_memory_;
    std::atomic<size_t> peak_memory_;
    std::vector<unsigned int> util_samples_;
    std::thread sampling_thread_;
    std::atomic<bool> stop_sampling_;
};
