# cuPQC Benchmark

This repository contains a complete benchmark suite for evaluating the performance of cuPQC (CUDA-based post-quantum cryptography) implementations. It includes tools for automated benchmarking of both Key Encapsulation Mechanisms (KEM) and Digital Signature Algorithms (DSA), as well as a comprehensive compatibility checker to ensure your system meets the necessary hardware and software requirements.

*Note: This project was inspired by and adapted from the work in the [Speed-Comparisons-cuPQC-Intel](https://github.com/lakshya-chopra/Speed-Comparisons-cuPQC-Intel) repository, with modifications for broader compatibility and extended functionality.*

## 📋 Prerequisites

Before using the benchmark suite, ensure the following prerequisites are met:

* **Operating System:** Ubuntu 22.04 or later (recommended)
* **CUDA:** Version 12.4 or later
* **NVIDIA Driver:** Compatible with your installed GPU
* **NVIDIA GPU:** A supported GPU with compute capability 7.5 or higher

## 📦 Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/pedrorigon/cuPQC_benchmark.git
cd cuPQC_benchmark
```

Ensure the main setup script is executable:

```bash
chmod +x setup_benchmark.sh
```

## 🚀 Usage

### 1. Compatibility Check (Optional)

Run the compatibility checker to ensure your system meets the minimum requirements:

```bash
./compatibility_check.py
```

or using Python directly:

```bash
python3 compatibility_check.py
```

### 2. Setup and Run the Benchmark Environment

Run the setup script to download, prepare, and execute the cuPQC benchmarks:

```bash
./setup_benchmark.sh
```

This will:

* Download the required cuPQC package.
* Move the benchmark scripts to the correct directory.
* Compile the benchmark executables.
* Run both the `benchmark_mlkem.sh` and `benchmark_mldsa.sh` scripts automatically.

### 3. Manual Execution (Optional)

If you want to run the benchmarks manually after the setup, navigate to the `benchmarks` directory:

```bash
cd cupqc/cupqc-pkg-0.2.0/benchmarks
./benchmark_mlkem.sh
./benchmark_mldsa.sh
```

## 📝 Output

The benchmark scripts will generate JSON files in the `output/` directory, containing the mean and standard deviation for each operation type:

### **KEM Output Example (`mlkem_results.json`)**

```json
{
  "GPU": {
    "name": "NVIDIA GeForce RTX 4090",
    "memory_total_mb":  24564
  },
  "ML-KEM-512": {
    "KeyGen": {"mean": 14050340.24, "std": 29988.03},
    "Encaps": {"mean": 15203110.36, "std": 41198.38},
    "Decaps": {"mean": 14113585.28, "std": 26719.60}
  },
  "ML-KEM-768": {
    "KeyGen": {"mean": 8983648.71, "std": 3121.49},
    "Encaps": {"mean": 8767379.22, "std": 4492.57},
    "Decaps": {"mean": 7761895.77, "std": 5079.65}
  },
  "ML-KEM-1024": {
    "KeyGen": {"mean": 5638323.14, "std": 2266.08},
    "Encaps": {"mean": 5336937.25, "std": 1728.08},
    "Decaps": {"mean": 5167359.88, "std": 2077.36}
  }
}
```

### **DSA Output Example (`ds_results.json`)**

```json
{
  "GPU": {
    "name": "NVIDIA GeForce RTX 4090",
    "memory_total_mb":  24564
  },
  "ML-DSA-44": {
    "KeyGen": {"mean": 9065289.41, "std": 13045.02},
    "Sign": {"mean": 1013504.21, "std": 1250.49},
    "Verify": {"mean": 6263178.58, "std": 12425.60}
  },
  "ML-DSA-65": {
    "KeyGen": {"mean": 3707841.33, "std": 6332.23},
    "Sign": {"mean": 773158.36, "std": 3473.93},
    "Verify": {"mean": 3065206.33, "std": 3284.52}
  },
  "ML-DSA-87": {
    "KeyGen": {"mean": 2783527.29, "std": 3463.64},
    "Sign": {"mean": 757954.25, "std": 2970.26},
    "Verify": {"mean": 2246405.81, "std": 1625.67}
  }
}
```

## 🔧 Troubleshooting

If you encounter errors related to missing drivers or unsupported architectures, ensure the following:

* **CUDA is correctly installed** (use `nvcc --version` to verify)
* **NVIDIA drivers are installed and up to date** (use `nvidia-smi` to verify)
* **Your GPU is in the supported list** (https://developer.nvidia.com/cuda-gpus)

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for more details.

## 📞 Support

For issues or questions, please open an issue on the GitHub repository or contact the maintainers directly.
