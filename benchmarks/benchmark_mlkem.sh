#!/usr/bin/env bash
set -euo pipefail

NUM_RUNS=1000
EXE="./bench_kem"
OUTDIR="output"
OUTFILE="$OUTDIR/data.json"

mkdir -p "$OUTDIR"

declare -a kg_512 enc_512 dec_512
declare -a kg_768 enc_768 dec_768
declare -a kg_1024 enc_1024 dec_1024

for i in $(seq 1 "$NUM_RUNS"); do
  echo -ne "Running $i/$NUM_RUNS\r"
  out="$($EXE)"

  kg_512+=( $(printf '%s\n' "$out" | sed -n 's/^\[ML-KEM-512\] KeyGen: \([0-9.]\+\).*/\1/p') )
  enc_512+=( $(printf '%s\n' "$out" | sed -n 's/^\[ML-KEM-512\] Encaps: \([0-9.]\+\).*/\1/p') )
  dec_512+=( $(printf '%s\n' "$out" | sed -n 's/^\[ML-KEM-512\] Decaps: \([0-9.]\+\).*/\1/p') )

  kg_768+=( $(printf '%s\n' "$out" | sed -n 's/^\[ML-KEM-768\] KeyGen: \([0-9.]\+\).*/\1/p') )
  enc_768+=( $(printf '%s\n' "$out" | sed -n 's/^\[ML-KEM-768\] Encaps: \([0-9.]\+\).*/\1/p') )
  dec_768+=( $(printf '%s\n' "$out" | sed -n 's/^\[ML-KEM-768\] Decaps: \([0-9.]\+\).*/\1/p') )

  kg_1024+=( $(printf '%s\n' "$out" | sed -n 's/^\[ML-KEM-1024\] KeyGen: \([0-9.]\+\).*/\1/p') )
  enc_1024+=( $(printf '%s\n' "$out" | sed -n 's/^\[ML-KEM-1024\] Encaps: \([0-9.]\+\).*/\1/p') )
  dec_1024+=( $(printf '%s\n' "$out" | sed -n 's/^\[ML-KEM-1024\] Decaps: \([0-9.]\+\).*/\1/p') )
done
echo ""

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
    "KeyGen": $(compute_json "${kg_512[@]}"),
    "Encaps": $(compute_json "${enc_512[@]}"),
    "Decaps": $(compute_json "${dec_512[@]}")
  },
  "ML-KEM-768": {
    "KeyGen": $(compute_json "${kg_768[@]}"),
    "Encaps": $(compute_json "${enc_768[@]}"),
    "Decaps": $(compute_json "${dec_768[@]}")
  },
  "ML-KEM-1024": {
    "KeyGen": $(compute_json "${kg_1024[@]}"),
    "Encaps": $(compute_json "${enc_1024[@]}"),
    "Decaps": $(compute_json "${dec_1024[@]}")
  }
}
EOF

echo "Written stats to $OUTFILE"
