#!/usr/bin/env bash
set -euo pipefail

NUM_RUNS=20
EXE="./bench_ds"
OUTDIR="output"
OUTFILE="$OUTDIR/ds_results.json"

mkdir -p "$OUTDIR"

declare -a kg_44_throughput kg_44_mem kg_44_gpu
declare -a sign_44_throughput sign_44_mem sign_44_gpu
declare -a verify_44_throughput verify_44_mem verify_44_gpu

declare -a kg_65_throughput kg_65_mem kg_65_gpu
declare -a sign_65_throughput sign_65_mem sign_65_gpu
declare -a verify_65_throughput verify_65_mem verify_65_gpu

declare -a kg_87_throughput kg_87_mem kg_87_gpu
declare -a sign_87_throughput sign_87_mem sign_87_gpu
declare -a verify_87_throughput verify_87_mem verify_87_gpu

for i in $(seq 1 "$NUM_RUNS"); do
  echo "Running $i/$NUM_RUNS"
  out="$($EXE)"
  
  # Extrai métricas específicas para cada operação/algoritmo
  kg_44_throughput+=( $(echo "$out" | sed -n '/ML-DSA-44 Key Generation/{n;s/^  Throughput: \([0-9.]\+\) ops\/sec/\1/p}') )
  kg_44_mem+=( $(echo "$out" | sed -n '/ML-DSA-44 Key Generation/{n;n;s/^  GPU Memory Used: \([0-9.]\+\) MB/\1/p}') )
  kg_44_gpu+=( $(echo "$out" | sed -n '/ML-DSA-44 Key Generation/{n;n;n;s/^  Peak GPU Utilization: \([0-9.]\+\)%/\1/p}') )
  
  sign_44_throughput+=( $(echo "$out" | sed -n '/ML-DSA-44 Signing/{n;s/^  Throughput: \([0-9.]\+\) ops\/sec/\1/p}') )
  sign_44_mem+=( $(echo "$out" | sed -n '/ML-DSA-44 Signing/{n;n;s/^  GPU Memory Used: \([0-9.]\+\) MB/\1/p}') )
  sign_44_gpu+=( $(echo "$out" | sed -n '/ML-DSA-44 Signing/{n;n;n;s/^  Peak GPU Utilization: \([0-9.]\+\)%/\1/p}') )
  
  verify_44_throughput+=( $(echo "$out" | sed -n '/ML-DSA-44 Verification/{n;s/^  Throughput: \([0-9.]\+\) ops\/sec/\1/p}') )
  verify_44_mem+=( $(echo "$out" | sed -n '/ML-DSA-44 Verification/{n;n;s/^  GPU Memory Used: \([0-9.]\+\) MB/\1/p}') )
  verify_44_gpu+=( $(echo "$out" | sed -n '/ML-DSA-44 Verification/{n;n;n;s/^  Peak GPU Utilization: \([0-9.]\+\)%/\1/p}') )
  
  kg_65_throughput+=( $(echo "$out" | sed -n '/ML-DSA-65 Key Generation/{n;s/^  Throughput: \([0-9.]\+\) ops\/sec/\1/p}') )
  kg_65_mem+=( $(echo "$out" | sed -n '/ML-DSA-65 Key Generation/{n;n;s/^  GPU Memory Used: \([0-9.]\+\) MB/\1/p}') )
  kg_65_gpu+=( $(echo "$out" | sed -n '/ML-DSA-65 Key Generation/{n;n;n;s/^  Peak GPU Utilization: \([0-9.]\+\)%/\1/p}') )
  
  sign_65_throughput+=( $(echo "$out" | sed -n '/ML-DSA-65 Signing/{n;s/^  Throughput: \([0-9.]\+\) ops\/sec/\1/p}') )
  sign_65_mem+=( $(echo "$out" | sed -n '/ML-DSA-65 Signing/{n;n;s/^  GPU Memory Used: \([0-9.]\+\) MB/\1/p}') )
  sign_65_gpu+=( $(echo "$out" | sed -n '/ML-DSA-65 Signing/{n;n;n;s/^  Peak GPU Utilization: \([0-9.]\+\)%/\1/p}') )
  
  verify_65_throughput+=( $(echo "$out" | sed -n '/ML-DSA-65 Verification/{n;s/^  Throughput: \([0-9.]\+\) ops\/sec/\1/p}') )
  verify_65_mem+=( $(echo "$out" | sed -n '/ML-DSA-65 Verification/{n;n;s/^  GPU Memory Used: \([0-9.]\+\) MB/\1/p}') )
  verify_65_gpu+=( $(echo "$out" | sed -n '/ML-DSA-65 Verification/{n;n;n;s/^  Peak GPU Utilization: \([0-9.]\+\)%/\1/p}') )
  
  kg_87_throughput+=( $(echo "$out" | sed -n '/ML-DSA-87 Key Generation/{n;s/^  Throughput: \([0-9.]\+\) ops\/sec/\1/p}') )
  kg_87_mem+=( $(echo "$out" | sed -n '/ML-DSA-87 Key Generation/{n;n;s/^  GPU Memory Used: \([0-9.]\+\) MB/\1/p}') )
  kg_87_gpu+=( $(echo "$out" | sed -n '/ML-DSA-87 Key Generation/{n;n;n;s/^  Peak GPU Utilization: \([0-9.]\+\)%/\1/p}') )
  
  sign_87_throughput+=( $(echo "$out" | sed -n '/ML-DSA-87 Signing/{n;s/^  Throughput: \([0-9.]\+\) ops\/sec/\1/p}') )
  sign_87_mem+=( $(echo "$out" | sed -n '/ML-DSA-87 Signing/{n;n;s/^  GPU Memory Used: \([0-9.]\+\) MB/\1/p}') )
  sign_87_gpu+=( $(echo "$out" | sed -n '/ML-DSA-87 Signing/{n;n;n;s/^  Peak GPU Utilization: \([0-9.]\+\)%/\1/p}') )
  
  verify_87_throughput+=( $(echo "$out" | sed -n '/ML-DSA-87 Verification/{n;s/^  Throughput: \([0-9.]\+\) ops\/sec/\1/p}') )
  verify_87_mem+=( $(echo "$out" | sed -n '/ML-DSA-87 Verification/{n;n;s/^  GPU Memory Used: \([0-9.]\+\) MB/\1/p}') )
  verify_87_gpu+=( $(echo "$out" | sed -n '/ML-DSA-87 Verification/{n;n;n;s/^  Peak GPU Utilization: \([0-9.]\+\)%/\1/p}') )
done

compute_json() {
  local arr=( "$@" )
  local sorted q1 q3 iqr lb ub filtered mean std n
  mapfile -t sorted < <(printf "%s\n" "${arr[@]}" | sort -n)
  n=${#sorted[@]}
  if (( n >= 4 )); then
    local i_q1=$(((n+3)/4))
    local i_q3=$(((3*n+5)/4))
    (( i_q1<1 ))  && i_q1=1
    (( i_q1>n ))  && i_q1=$n
    (( i_q3<1 ))  && i_q3=$n
    q1=${sorted[i_q1-1]}; q3=${sorted[i_q3-1]}
    iqr=$(echo "$q3 - $q1" | bc -l)
    lb=$(echo "$q1 - 1.5*$iqr" | bc -l)
    ub=$(echo "$q3 + 1.5*$iqr" | bc -l)
    filtered=()
    for v in "${sorted[@]}"; do
      if (( $(echo "$v >= $lb && $v <= $ub" | bc -l) )); then
        filtered+=( "$v" )
      fi
    done
    (( ${#filtered[@]} == 0 )) && filtered=( "${sorted[@]}" )
  else
    filtered=( "${sorted[@]}" )
  fi

  awk 'BEGIN{sum=0;sum2=0;n=0;}
    { sum+=$1; sum2+=$1*$1; n++ }
    END {
      mean=sum/n;
      std=sqrt(sum2/n - mean*mean);
      printf("{\"mean\": %.2f, \"std\": %.2f}", mean, std);
    }' < <(printf "%s\n" "${filtered[@]}")
}

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -n 1)
IFS=',' read -r gpu_name gpu_memory_total <<< "$GPU_INFO"

cat > "$OUTFILE" <<EOF
{
  "GPU": {
    "name": "$gpu_name",
    "memory_total_mb": $gpu_memory_total
  },
  "ML-DSA-44": {
    "KeyGen": {
      "throughput": $(compute_json "${kg_44_throughput[@]}"),
      "peak_mem_mb": $(compute_json "${kg_44_mem[@]}"),
      "peak_gpu_util": $(compute_json "${kg_44_gpu[@]}")
    },
    "Sign": {
      "throughput": $(compute_json "${sign_44_throughput[@]}"),
      "peak_mem_mb": $(compute_json "${sign_44_mem[@]}"),
      "peak_gpu_util": $(compute_json "${sign_44_gpu[@]}")
    },
    "Verify": {
      "throughput": $(compute_json "${verify_44_throughput[@]}"),
      "peak_mem_mb": $(compute_json "${verify_44_mem[@]}"),
      "peak_gpu_util": $(compute_json "${verify_44_gpu[@]}")
    }
  },
  "ML-DSA-65": {
    "KeyGen": {
      "throughput": $(compute_json "${kg_65_throughput[@]}"),
      "peak_mem_mb": $(compute_json "${kg_65_mem[@]}"),
      "peak_gpu_util": $(compute_json "${kg_65_gpu[@]}")
    },
    "Sign": {
      "throughput": $(compute_json "${sign_65_throughput[@]}"),
      "peak_mem_mb": $(compute_json "${sign_65_mem[@]}"),
      "peak_gpu_util": $(compute_json "${sign_65_gpu[@]}")
    },
    "Verify": {
      "throughput": $(compute_json "${verify_65_throughput[@]}"),
      "peak_mem_mb": $(compute_json "${verify_65_mem[@]}"),
      "peak_gpu_util": $(compute_json "${verify_65_gpu[@]}")
    }
  },
  "ML-DSA-87": {
    "KeyGen": {
      "throughput": $(compute_json "${kg_87_throughput[@]}"),
      "peak_mem_mb": $(compute_json "${kg_87_mem[@]}"),
      "peak_gpu_util": $(compute_json "${kg_87_gpu[@]}")
    },
    "Sign": {
      "throughput": $(compute_json "${sign_87_throughput[@]}"),
      "peak_mem_mb": $(compute_json "${sign_87_mem[@]}"),
      "peak_gpu_util": $(compute_json "${sign_87_gpu[@]}")
    },
    "Verify": {
      "throughput": $(compute_json "${verify_87_throughput[@]}"),
      "peak_mem_mb": $(compute_json "${verify_87_mem[@]}"),
      "peak_gpu_util": $(compute_json "${verify_87_gpu[@]}")
    }
  }
}
EOF

echo "Written stats to $OUTFILE"