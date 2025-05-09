#!/usr/bin/env python3
import subprocess
import sys
import platform
import re

REQUIRED_CUDA_VERSION = (12, 4)
SUPPORTED_ARCHITECTURES = {
    120: ["GeForce RTX 5090", "GeForce RTX 5080", "GeForce RTX 5070 Ti", "GeForce RTX 5070"],
    100: ["NVIDIA GB200", "NVIDIA B200"],
    90: ["NVIDIA GH200", "NVIDIA H200", "NVIDIA H100"],
    89: ["NVIDIA L4", "NVIDIA L40", "RTX 6000 Ada", "RTX 5000 Ada", "RTX 4500 Ada", "RTX 4000 Ada", "RTX 4000 SFF Ada", "RTX 2000 Ada", "GeForce RTX 4090", "GeForce RTX 4080", "GeForce RTX 4070 Ti", "GeForce RTX 4070", "GeForce RTX 4060 Ti", "GeForce RTX 4060", "GeForce RTX 4050"],
    87: ["Jetson AGX Orin", "Jetson Orin NX", "Jetson Orin Nano"],
    86: ["NVIDIA A40", "NVIDIA A10", "NVIDIA A16", "NVIDIA A2", "RTX A6000", "RTX A5000", "RTX A4000", "RTX A3000", "RTX A2000", "GeForce RTX 3090 Ti", "GeForce RTX 3090", "GeForce RTX 3080 Ti", "GeForce RTX 3080", "GeForce RTX 3070 Ti", "GeForce RTX 3070", "GeForce RTX 3060 Ti", "GeForce RTX 3060", "GeForce RTX 3050 Ti", "GeForce RTX 3050"],
    80: ["NVIDIA A100", "NVIDIA A30"],
    75: ["NVIDIA T4", "T1000", "T600", "T400", "T2000", "T1200", "T500", "RTX 8000", "RTX 6000", "RTX 5000", "RTX 4000", "RTX 3000", "GeForce GTX 1650 Ti", "NVIDIA TITAN RTX", "GeForce RTX 2080 Ti", "GeForce RTX 2080", "GeForce RTX 2070", "GeForce RTX 2060"]
}

def check_cpu_arch():
    arch = platform.machine()
    if arch == "x86_64":
        print("[✔] Compatible CPU architecture (x86_64) detected.")
        return True
    else:
        print(f"[✘] Incompatible CPU architecture ({arch}) detected. Expected x86_64.")
        return False

def check_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT).decode()
        match = re.search(r"release (\d+)\.(\d+)", output)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            if (major, minor) >= REQUIRED_CUDA_VERSION:
                print(f"[✔] Compatible CUDA version detected: {major}.{minor}")
                return True
            else:
                print(f"[✘] Incompatible CUDA version detected: {major}.{minor}. Requires {REQUIRED_CUDA_VERSION[0]}.{REQUIRED_CUDA_VERSION[1]} or newer.")
                print("Please install the correct version. You can try running ./install_cuda.sh to set it up.")
                return False
        else:
            print("[✘] Could not determine CUDA version. Is nvcc installed?")
            return False
    except FileNotFoundError:
        print(f"[✘] CUDA not found. Please install CUDA {REQUIRED_CUDA_VERSION[0]}.{REQUIRED_CUDA_VERSION[1]} or newer. Try running ./install_cuda.sh")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[✘] Error checking CUDA version: {e.output.decode().strip()}")
        return False

def check_gpu_arch():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], stderr=subprocess.STDOUT).decode().splitlines()
        compatible = False

        for gpu_name in output:
            found = False
            for capability, models in SUPPORTED_ARCHITECTURES.items():
                if any(model in gpu_name for model in models):
                    print(f"[✔] Compatible GPU detected: {gpu_name} (Compute Capability {capability})")
                    found = True
                    compatible = True
                    break
            if not found:
                print(f"[✘] Incompatible GPU detected: {gpu_name}")

        return compatible

    except subprocess.CalledProcessError as e:
        print(f"[✘] Error checking GPU architecture: {e.output.decode().strip()}")
        return False
    except FileNotFoundError:
        print("[✘] NVIDIA GPU not found. Ensure the correct drivers are installed.")
        return False
    except Exception as e:
        print(f"[✘] Unexpected error checking GPU architecture: {e}")
        return False

def main():
    print("Running compatibility check...")
    cpu_ok = check_cpu_arch()
    cuda_ok = check_cuda_version()
    gpu_ok = check_gpu_arch()

    if cpu_ok and cuda_ok and gpu_ok:
        print("\n[✔] System is compatible for setup.")
        sys.exit(0)
    else:
        print("\n[✘] System is not compatible for setup. Please address the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
