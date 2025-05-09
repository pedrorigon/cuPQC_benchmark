#!/usr/bin/env bash
set -euo pipefail

NUM_RUNS=1000
EXE="./bench_ds"
OUTDIR="output"
OUTFILE="$OUTDIR/ds_results.json"

mkdir -p "$OUTDIR"

declare -a kg_44 sign_44 verify_44
declare -a kg_65 sign_65 verify_65
declare -a kg_87 sign_87 verify_87

for i in $(seq 1 "$NUM_RUNS"); do
  echo "Running $i/$NUM_RUNS"
  out="$($EXE)"

  kg_44+=( $(printf '%s\n' "$out" | sed -n 's/^ML-DSA-44 Key Generation Throughput: \([0-9.]\+\) ops\/sec/\1/p') )
  sign_44+=( $(printf '%s\n' "$out" | sed -n 's/^ML-DSA-44 Signing Throughput: \([0-9.]\+\) ops\/sec/\1/p') )
  verify_44+=( $(printf '%s\n' "$out" | sed -n 's/^ML-DSA-44 Verification Throughput: \([0-9.]\+\) ops\/sec/\1/p') )

  kg_65+=( $(printf '%s\n' "$out" | sed -n 's/^ML-DSA-65 Key Generation Throughput: \([0-9.]\+\) ops\/sec/\1/p') )
  sign_65+=( $(printf '%s\n' "$out" | sed -n 's/^ML-DSA-65 Signing Throughput: \([0-9.]\+\) ops\/sec/\1/p') )
  verify_65+=( $(printf '%s\n' "$out" | sed -n 's/^ML-DSA-65 Verification Throughput: \([0-9.]\+\) ops\/sec/\1/p') )

  kg_87+=( $(printf '%s\n' "$out" | sed -n 's/^ML-DSA-87 Key Generation Throughput: \([0-9.]\+\) ops\/sec/\1/p') )
  sign_87+=( $(printf '%s\n' "$out" | sed -n 's/^ML-DSA-87 Signing Throughput: \([0-9.]\+\) ops\/sec/\1/p') )
  verify_87+=( $(printf '%s\n' "$out" | sed -n 's/^ML-DSA-87 Verification Throughput: \([0-9.]\+\) ops\/sec/\1/p') )
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
    "KeyGen": $(compute_json "${kg_44[@]}"),
    "Sign": $(compute_json "${sign_44[@]}"),
    "Verify": $(compute_json "${verify_44[@]}")
  },
  "ML-DSA-65": {
    "KeyGen": $(compute_json "${kg_65[@]}"),
    "Sign": $(compute_json "${sign_65[@]}"),
    "Verify": $(compute_json "${verify_65[@]}")
  },
  "ML-DSA-87": {
    "KeyGen": $(compute_json "${kg_87[@]}"),
    "Sign": $(compute_json "${sign_87[@]}"),
    "Verify": $(compute_json "${verify_87[@]}")
  }
}
EOF

echo "Written stats to $OUTFILE"
