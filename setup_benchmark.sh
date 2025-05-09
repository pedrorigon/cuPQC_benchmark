#!/bin/bash
set -euo pipefail

if ! nvcc --version | grep -q "release 12.4"; then
    echo "CUDA 12.4 or newer is required. Please install the correct version."
    exit 1
fi

wget https://developer.download.nvidia.com/compute/cupqc/redist/cupqc/cupqc-pkg-0.2.0.tar.gz
tar -xvzf cupqc-pkg-0.2.0.tar.gz
cp -r benchmarks cupqc-pkg-0.2.0/
cd cupqc-pkg-0.2.0/benchmarks
make
chmod +x benchmark_mlkem.sh
./benchmark_mlkem.sh
