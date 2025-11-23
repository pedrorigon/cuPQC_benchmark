#!/bin/bash
set -euo pipefail

REQUIRED_VERSION="12.4"
MODE="${1:-normal}"

check_cuda() {
    local nvcc_version
    nvcc_version=$(nvcc --version 2>/dev/null | grep "release" | sed -E 's/.*release ([0-9]+\.[0-9]+).*/\1/' || echo "")

    if [[ -z "$nvcc_version" ]] || ! awk -v ver="$nvcc_version" -v req="$REQUIRED_VERSION" 'BEGIN { if (ver < req) exit 1; exit 0 }'; then
        echo "CUDA $REQUIRED_VERSION or newer is required. Please install the correct version. You can try running ./install_cuda.sh to set it up."
        exit 1
    fi
}

run_benchmarks() {
    cd benchmarks
    make

    chmod +x benchmark_mlkem.sh benchmark_mldsa.sh \
             extract_latency_mlkem.sh extract_latency_mldsa.sh

    if [ "$MODE" = "latency" ]; then
        echo "Running latency extraction..."
        ./extract_latency_mlkem.sh
        ./extract_latency_mldsa.sh
    else
        echo "Running normal benchmarks..."
        ./benchmark_mlkem.sh
        ./benchmark_mldsa.sh
    fi
}

check_cuda

if [ -d "dependencies/cupqc-pkg-0.2.0" ]; then
    echo "Dependencies already present."
else
    mkdir -p dependencies
    wget https://developer.download.nvidia.com/compute/cupqc/redist/cupqc/cupqc-pkg-0.2.0.tar.gz
    tar -xvzf cupqc-pkg-0.2.0.tar.gz -C dependencies/ --strip-components=1
    rm cupqc-pkg-0.2.0.tar.gz

    rm -rf dependencies/cupqc-pkg-0.2.0/benchmarks
fi

run_benchmarks
