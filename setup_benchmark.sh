#!/bin/bash
set -euo pipefail

REQUIRED_VERSION="12.4"

nvcc_version=$(nvcc --version 2>/dev/null | grep "release" | sed -E 's/.*release ([0-9]+\.[0-9]+).*/\1/' || echo "")

if [[ -z "$nvcc_version" ]] || ! awk -v ver="$nvcc_version" -v req="$REQUIRED_VERSION" 'BEGIN { if (ver < req) exit 1; exit 0 }'; then
    echo "CUDA $REQUIRED_VERSION or newer is required. Please install the correct version. You can try running ./install_cuda.sh to set it up."
    exit 1
fi

if [ -d "dependencies/cupqc-pkg-0.2.0" ]; then
    echo "Dependencies already present."
    cd benchmarks
    chmod +x benchmark_mlkem.sh benchmark_mldsa.sh
    # ./benchmark_mlkem.sh
    # ./benchmark_mldsa.sh
    exit 0
fi

mkdir -p dependencies
wget https://developer.download.nvidia.com/compute/cupqc/redist/cupqc/cupqc-pkg-0.2.0.tar.gz
tar -xvzf cupqc-pkg-0.2.0.tar.gz -C dependencies/ --strip-components=1
rm cupqc-pkg-0.2.0.tar.gz

# Remove the benchmarks folder from dependencies to avoid confusion/duplication
rm -rf dependencies/cupqc-pkg-0.2.0/benchmarks

cd benchmarks
make
chmod +x benchmark_mlkem.sh benchmark_mldsa.sh
# ./benchmark_mlkem.sh
# ./benchmark_mldsa.sh
