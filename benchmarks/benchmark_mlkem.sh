#!/usr/bin/env bash
set -euo pipefail
export LC_NUMERIC=C

NUM_RUNS=20
EXE="./bench_kem"
OUTDIR="../outputs"
OUTFILE="$OUTDIR/mlkem_results.json"

mkdir -p "$OUTDIR"

declare -a kg_512_throughput kg_512_mem kg_512_gpu
declare -a enc_512_throughput enc_512_mem enc_512_gpu
declare -a dec_512_throughput dec_512_mem dec_512_gpu

declare -a kg_768_throughput kg_768_mem kg_768_gpu
declare -a enc_768_throughput enc_768_mem enc_768_gpu
declare -a dec_768_throughput dec_768_mem dec_768_gpu

declare -a kg_1024_throughput kg_1024_mem kg_1024_gpu
declare -a enc_1024_throughput enc_1024_mem enc_1024_gpu
declare -a dec_1024_throughput dec_1024_mem dec_1024_gpu

for i in $(seq 1 "$NUM_RUNS"); do
  echo "Running $i/$NUM_RUNS"
  out="$($EXE)"

  # ML-KEM-512
  kg_512_throughput+=( $(echo "$out" | sed -n '/ML-KEM-512 KeyGen/{n;s/^  Throughput: \([0-9.]\+\) ops\/sec/\1/p}') )
  kg_512_mem+=( $(echo "$out" | sed -n '/ML-KEM-512 KeyGen/{n;n;s/^  Peak GPU Memory Used: \([0-9.]\+\) MB/\1/p}') )
  kg_512_gpu+=( $(echo "$out" | sed -n '/ML-KEM-512 KeyGen/{n;n;n;s/^  Peak GPU Utilization: \([0-9.]\+\)%/\1/p}') )

  enc_512_throughput+=( $(echo "$out" | sed -n '/ML-KEM-512 Encaps/{n;s/^  Throughput: \([0-9.]\+\) ops\/sec/\1/p}') )
  enc_512_mem+=( $(echo "$out" | sed -n '/ML-KEM-512 Encaps/{n;n;s/^  Peak GPU Memory Used: \([0-9.]\+\) MB/\1/p}') )
  enc_512_gpu+=( $(echo "$out" | sed -n '/ML-KEM-512 Encaps/{n;n;n;s/^  Peak GPU Utilization: \([0-9.]\+\)%/\1/p}') )

  dec_512_throughput+=( $(echo "$out" | sed -n '/ML-KEM-512 Decaps/{n;s/^  Throughput: \([0-9.]\+\) ops\/sec/\1/p}') )
  dec_512_mem+=( $(echo "$out" | sed -n '/ML-KEM-512 Decaps/{n;n;s/^  Peak GPU Memory Used: \([0-9.]\+\) MB/\1/p}') )
  dec_512_gpu+=( $(echo "$out" | sed -n '/ML-KEM-512 Decaps/{n;n;n;s/^  Peak GPU Utilization: \([0-9.]\+\)%/\1/p}') )

  # ML-KEM-768
  kg_768_throughput+=( $(echo "$out" | sed -n '/ML-KEM-768 KeyGen/{n;s/^  Throughput: \([0-9.]\+\) ops\/sec/\1/p}') )
  kg_768_mem+=( $(echo "$out" | sed -n '/ML-KEM-768 KeyGen/{n;n;s/^  Peak GPU Memory Used: \([0-9.]\+\) MB/\1/p}') )
  kg_768_gpu+=( $(echo "$out" | sed -n '/ML-KEM-768 KeyGen/{n;n;n;s/^  Peak GPU Utilization: \([0-9.]\+\)%/\1/p}') )

  enc_768_throughput+=( $(echo "$out" | sed -n '/ML-KEM-768 Encaps/{n;s/^  Throughput: \([0-9.]\+\) ops\/sec/\1/p}') )
  enc_768_mem+=( $(echo "$out" | sed -n '/ML-KEM-768 Encaps/{n;n;s/^  Peak GPU Memory Used: \([0-9.]\+\) MB/\1/p}') )
  enc_768_gpu+=( $(echo "$out" | sed -n '/ML-KEM-768 Encaps/{n;n;n;s/^  Peak GPU Utilization: \([0-9.]\+\)%/\1/p}') )

  dec_768_throughput+=( $(echo "$out" | sed -n '/ML-KEM-768 Decaps/{n;s/^  Throughput: \([0-9.]\+\) ops\/sec/\1/p}') )
  dec_768_mem+=( $(echo "$out" | sed -n '/ML-KEM-768 Decaps/{n;n;s/^  Peak GPU Memory Used: \([0-9.]\+\) MB/\1/p}') )
  dec_768_gpu+=( $(echo "$out" | sed -n '/ML-KEM-768 Decaps/{n;n;n;s/^  Peak GPU Utilization: \([0-9.]\+\)%/\1/p}') )

  # ML-KEM-1024
  kg_1024_throughput+=( $(echo "$out" | sed -n '/ML-KEM-1024 KeyGen/{n;s/^  Throughput: \([0-9.]\+\) ops\/sec/\1/p}') )
  kg_1024_mem+=( $(echo "$out" | sed -n '/ML-KEM-1024 KeyGen/{n;n;s/^  Peak GPU Memory Used: \([0-9.]\+\) MB/\1/p}') )
  kg_1024_gpu+=( $(echo "$out" | sed -n '/ML-KEM-1024 KeyGen/{n;n;n;s/^  Peak GPU Utilization: \([0-9.]\+\)%/\1/p}') )

  enc_1024_throughput+=( $(echo "$out" | sed -n '/ML-KEM-1024 Encaps/{n;s/^  Throughput: \([0-9.]\+\) ops\/sec/\1/p}') )
  enc_1024_mem+=( $(echo "$out" | sed -n '/ML-KEM-1024 Encaps/{n;n;s/^  Peak GPU Memory Used: \([0-9.]\+\) MB/\1/p}') )
  enc_1024_gpu+=( $(echo "$out" | sed -n '/ML-KEM-1024 Encaps/{n;n;n;s/^  Peak GPU Utilization: \([0-9.]\+\)%/\1/p}') )

  dec_1024_throughput+=( $(echo "$out" | sed -n '/ML-KEM-1024 Decaps/{n;s/^  Throughput: \([0-9.]\+\) ops\/sec/\1/p}') )
  dec_1024_mem+=( $(echo "$out" | sed -n '/ML-KEM-1024 Decaps/{n;n;s/^  Peak GPU Memory Used: \([0-9.]\+\) MB/\1/p}') )
  dec_1024_gpu+=( $(echo "$out" | sed -n '/ML-KEM-1024 Decaps/{n;n;n;s/^  Peak GPU Utilization: \([0-9.]\+\)%/\1/p}') )
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
    (( i_q3<1 ))  && i_q3=1
    (( i_q3>n ))  && i_q3=$n
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
  "ML-KEM-512": {
    "KeyGen": {
      "throughput": $(compute_json "${kg_512_throughput[@]}"),
      "peak_mem_mb": $(compute_json "${kg_512_mem[@]}"),
      "peak_gpu_util": $(compute_json "${kg_512_gpu[@]}")
    },
    "Encaps": {
      "throughput": $(compute_json "${enc_512_throughput[@]}"),
      "peak_mem_mb": $(compute_json "${enc_512_mem[@]}"),
      "peak_gpu_util": $(compute_json "${enc_512_gpu[@]}")
    },
    "Decaps": {
      "throughput": $(compute_json "${dec_512_throughput[@]}"),
      "peak_mem_mb": $(compute_json "${dec_512_mem[@]}"),
      "peak_gpu_util": $(compute_json "${dec_512_gpu[@]}")
    }
  },
  "ML-KEM-768": {
    "KeyGen": {
      "throughput": $(compute_json "${kg_768_throughput[@]}"),
      "peak_mem_mb": $(compute_json "${kg_768_mem[@]}"),
      "peak_gpu_util": $(compute_json "${kg_768_gpu[@]}")
    },
    "Encaps": {
      "throughput": $(compute_json "${enc_768_throughput[@]}"),
      "peak_mem_mb": $(compute_json "${enc_768_mem[@]}"),
      "peak_gpu_util": $(compute_json "${enc_768_gpu[@]}")
    },
    "Decaps": {
      "throughput": $(compute_json "${dec_768_throughput[@]}"),
      "peak_mem_mb": $(compute_json "${dec_768_mem[@]}"),
      "peak_gpu_util": $(compute_json "${dec_768_gpu[@]}")
    }
  },
  "ML-KEM-1024": {
    "KeyGen": {
      "throughput": $(compute_json "${kg_1024_throughput[@]}"),
      "peak_mem_mb": $(compute_json "${kg_1024_mem[@]}"),
      "peak_gpu_util": $(compute_json "${kg_1024_gpu[@]}")
    },
    "Encaps": {
      "throughput": $(compute_json "${enc_1024_throughput[@]}"),
      "peak_mem_mb": $(compute_json "${enc_1024_mem[@]}"),
      "peak_gpu_util": $(compute_json "${enc_1024_gpu[@]}")
    },
    "Decaps": {
      "throughput": $(compute_json "${dec_1024_throughput[@]}"),
      "peak_mem_mb": $(compute_json "${dec_1024_mem[@]}"),
      "peak_gpu_util": $(compute_json "${dec_1024_gpu[@]}")
    }
  }
}
EOF

echo "Written stats to $OUTFILE"
