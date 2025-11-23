#!/usr/bin/env bash
set -euo pipefail
export LC_NUMERIC=C

NUM_RUNS=20
EXE="./bench_ds_latency"
OUTDIR="../outputs/latency"
OUTFILE="$OUTDIR/mldsa_latency.json"

mkdir -p "$OUTDIR"

declare -a kg_44_latency sign_44_latency verify_44_latency
declare -a kg_65_latency sign_65_latency verify_65_latency
declare -a kg_87_latency sign_87_latency verify_87_latency

for i in $(seq 1 "$NUM_RUNS"); do
  echo "Running latency test $i/$NUM_RUNS"
  out="$($EXE)"

  # ML-DSA-44
  kg_44_latency+=( $(echo "$out" | sed -n 's/^ML-DSA-44 Key Generation: \([0-9.]\+\)/\1/p') )
  sign_44_latency+=( $(echo "$out" | sed -n 's/^ML-DSA-44 Signing: \([0-9.]\+\)/\1/p') )
  verify_44_latency+=( $(echo "$out" | sed -n 's/^ML-DSA-44 Verification: \([0-9.]\+\)/\1/p') )

  # ML-DSA-65
  kg_65_latency+=( $(echo "$out" | sed -n 's/^ML-DSA-65 Key Generation: \([0-9.]\+\)/\1/p') )
  sign_65_latency+=( $(echo "$out" | sed -n 's/^ML-DSA-65 Signing: \([0-9.]\+\)/\1/p') )
  verify_65_latency+=( $(echo "$out" | sed -n 's/^ML-DSA-65 Verification: \([0-9.]\+\)/\1/p') )

  # ML-DSA-87
  kg_87_latency+=( $(echo "$out" | sed -n 's/^ML-DSA-87 Key Generation: \([0-9.]\+\)/\1/p') )
  sign_87_latency+=( $(echo "$out" | sed -n 's/^ML-DSA-87 Signing: \([0-9.]\+\)/\1/p') )
  verify_87_latency+=( $(echo "$out" | sed -n 's/^ML-DSA-87 Verification: \([0-9.]\+\)/\1/p') )
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
      printf("{\"mean\": %.4f, \"std\": %.4f}", mean, std);
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
    "Key Generation": {
      "latency_us": $(compute_json "${kg_44_latency[@]}")
    },
    "Signing": {
      "latency_us": $(compute_json "${sign_44_latency[@]}")
    },
    "Verification": {
      "latency_us": $(compute_json "${verify_44_latency[@]}")
    }
  },
  "ML-DSA-65": {
    "Key Generation": {
      "latency_us": $(compute_json "${kg_65_latency[@]}")
    },
    "Signing": {
      "latency_us": $(compute_json "${sign_65_latency[@]}")
    },
    "Verification": {
      "latency_us": $(compute_json "${verify_65_latency[@]}")
    }
  },
  "ML-DSA-87": {
    "Key Generation": {
      "latency_us": $(compute_json "${kg_87_latency[@]}")
    },
    "Signing": {
      "latency_us": $(compute_json "${sign_87_latency[@]}")
    },
    "Verification": {
      "latency_us": $(compute_json "${verify_87_latency[@]}")
    }
  }
}
EOF

echo "Written latency stats to $OUTFILE"
