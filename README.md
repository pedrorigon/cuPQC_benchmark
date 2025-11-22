# cuPQC Benchmark

A comprehensive benchmarking suite for evaluating CUDA-accelerated post-quantum cryptographic implementations, specifically ML-KEM (Key Encapsulation Mechanism) and ML-DSA (Digital Signature Algorithm).

## Overview

This project provides automated tools for performance benchmarking of cuPQC implementations, measuring throughput, GPU memory usage, and utilization metrics across different security levels. The suite is designed for researchers and developers working on post-quantum cryptography performance analysis.

## Features

- Automated benchmarking for ML-KEM-512/768/1024 and ML-DSA-44/65/87
- GPU resource profiling (memory usage and utilization)
- Statistical analysis with outlier filtering using IQR method
- JSON-formatted results for easy integration with analysis pipelines
- Compatibility checking for CUDA and GPU requirements

## Prerequisites

- **Operating System**: Ubuntu 22.04 or later
- **CUDA**: Version 12.4 or later
- **NVIDIA Driver**: Compatible with CUDA 12.4+
- **NVIDIA GPU**: Compute capability 7.5 or higher
- **Dependencies**: NVML (NVIDIA Management Library)

## Installation

Clone the repository:

```bash
git clone https://github.com/pedrorigon/cuPQC_benchmark.git
cd cuPQC_benchmark
```

Make scripts executable:

```bash
chmod +x setup_benchmark.sh clean_build.sh
chmod +x compatibility_check.py
```

## Usage

### Compatibility Check

Verify system requirements before running benchmarks:

```bash
./compatibility_check.py
```

### Setup and Execution

Download dependencies, compile benchmarks, and run:

```bash
./setup_benchmark.sh
```

This script will:
1. Download cuPQC package to `dependencies/`
2. Compile benchmark executables
3. Execute benchmark scripts
4. Save results to `outputs/`

### Manual Execution

To run benchmarks manually:

```bash
cd benchmarks
./benchmark_mlkem.sh
./benchmark_mldsa.sh
```

### Clean Build Artifacts

Restore repository to clean state:

```bash
./clean_build.sh
```

## Output Format

Benchmark results are saved as JSON files in the `outputs/` directory.

### ML-KEM Results (mlkem_results.json)

```json
{
  "GPU": {
    "name": "NVIDIA GeForce RTX 3070",
    "memory_total_mb": 8192
  },
  "ML-KEM-512": {
    "KeyGen": {
      "throughput": {"mean": 14050340.24, "std": 29988.03},
      "peak_mem_mb": {"mean": 1616.00, "std": 7.00},
      "peak_gpu_util": {"mean": 24.50, "std": 13.00}
    },
    "Encaps": {
      "throughput": {"mean": 15203110.36, "std": 41198.38},
      "peak_mem_mb": {"mean": 1657.00, "std": 14.00},
      "peak_gpu_util": {"mean": 21.00, "std": 15.00}
    },
    "Decaps": {
      "throughput": {"mean": 14113585.28, "std": 26719.60},
      "peak_mem_mb": {"mean": 2138.00, "std": 7.00},
      "peak_gpu_util": {"mean": 40.50, "std": 16.00}
    }
  },
  "ML-KEM-768": {
    "KeyGen": {
      "throughput": {"mean": 8983648.71, "std": 3121.49},
      "peak_mem_mb": {"mean": 1925.00, "std": 15.00},
      "peak_gpu_util": {"mean": 35.00, "std": 19.00}
    },
    "Encaps": {
      "throughput": {"mean": 8767379.22, "std": 4492.57},
      "peak_mem_mb": {"mean": 2149.00, "std": 15.00},
      "peak_gpu_util": {"mean": 24.00, "std": 18.00}
    },
    "Decaps": {
      "throughput": {"mean": 7761895.77, "std": 5079.65},
      "peak_mem_mb": {"mean": 2672.00, "std": 14.00},
      "peak_gpu_util": {"mean": 78.00, "std": 17.00}
    }
  },
  "ML-KEM-1024": {
    "KeyGen": {
      "throughput": {"mean": 5638323.14, "std": 2266.08},
      "peak_mem_mb": {"mean": 2531.00, "std": 16.00},
      "peak_gpu_util": {"mean": 15.00, "std": 14.00}
    },
    "Encaps": {
      "throughput": {"mean": 5336937.25, "std": 1728.08},
      "peak_mem_mb": {"mean": 2822.00, "std": 16.00},
      "peak_gpu_util": {"mean": 52.00, "std": 24.00}
    },
    "Decaps": {
      "throughput": {"mean": 5167359.88, "std": 2077.36},
      "peak_mem_mb": {"mean": 3627.00, "std": 16.00},
      "peak_gpu_util": {"mean": 90.00, "std": 3.00}
    }
  }
}
```

### ML-DSA Results (mldsa_results.json)

```json
{
  "GPU": {
    "name": "NVIDIA GeForce RTX 3070",
    "memory_total_mb": 8192
  },
  "ML-DSA-44": {
    "KeyGen": {
      "throughput": {"mean": 9065289.41, "std": 13045.02},
      "peak_mem_mb": {"mean": 1305.00, "std": 19.00},
      "peak_gpu_util": {"mean": 66.50, "std": 12.00}
    },
    "Sign": {
      "throughput": {"mean": 1013504.21, "std": 1250.49},
      "peak_mem_mb": {"mean": 1680.00, "std": 19.00},
      "peak_gpu_util": {"mean": 13.00, "std": 2.00}
    },
    "Verify": {
      "throughput": {"mean": 6263178.58, "std": 12425.60},
      "peak_mem_mb": {"mean": 1634.00, "std": 19.00},
      "peak_gpu_util": {"mean": 13.20, "std": 2.00}
    }
  },
  "ML-DSA-65": {
    "KeyGen": {
      "throughput": {"mean": 3707841.33, "std": 6332.23},
      "peak_mem_mb": {"mean": 1489.00, "std": 24.00},
      "peak_gpu_util": {"mean": 18.00, "std": 13.00}
    },
    "Sign": {
      "throughput": {"mean": 773158.36, "std": 3473.93},
      "peak_mem_mb": {"mean": 2001.00, "std": 24.00},
      "peak_gpu_util": {"mean": 46.00, "std": 17.00}
    },
    "Verify": {
      "throughput": {"mean": 3065206.33, "std": 3284.52},
      "peak_mem_mb": {"mean": 1779.00, "std": 23.00},
      "peak_gpu_util": {"mean": 58.00, "std": 13.00}
    }
  },
  "ML-DSA-87": {
    "KeyGen": {
      "throughput": {"mean": 2783527.29, "std": 3463.64},
      "peak_mem_mb": {"mean": 1939.00, "std": 23.00},
      "peak_gpu_util": {"mean": 63.00, "std": 9.00}
    },
    "Sign": {
      "throughput": {"mean": 757954.25, "std": 2970.26},
      "peak_mem_mb": {"mean": 2643.00, "std": 23.00},
      "peak_gpu_util": {"mean": 66.00, "std": 5.00}
    },
    "Verify": {
      "throughput": {"mean": 2246405.81, "std": 1625.67},
      "peak_mem_mb": {"mean": 2153.00, "std": 24.00},
      "peak_gpu_util": {"mean": 65.00, "std": 2.00}
    }
  }
}
```

## Metrics Explained

- **throughput**: Operations per second (mean and standard deviation)
- **peak_mem_mb**: Maximum GPU memory usage in MB during operation
- **peak_gpu_util**: Peak GPU utilization percentage

Results are averaged over 20 runs with outlier filtering using the Interquartile Range (IQR) method.

## Citation

If you use this benchmark suite in your research, please cite:

```bibtex
@software{rigon2024cupqc_benchmark,
  author = {Rigon, Pedro},
  title = {cuPQC Benchmark: Performance Evaluation Suite for CUDA Post-Quantum Cryptography},
  year = {2024},
  url = {https://github.com/pedrorigon/cuPQC_benchmark},
  note = {Benchmark suite for ML-KEM and ML-DSA implementations using NVIDIA cuPQC}
}
```

## Acknowledgments

This project was inspired by and adapted from [Speed-Comparisons-cuPQC-Intel](https://github.com/lakshya-chopra/Speed-Comparisons-cuPQC-Intel) with modifications for extended functionality and broader GPU compatibility.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

For issues or questions, please open an issue on the [GitHub repository](https://github.com/pedrorigon/cuPQC_benchmark).
