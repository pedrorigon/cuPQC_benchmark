#!/bin/bash

echo "Cleaning build artifacts..."

rm -f benchmarks/bench_kem benchmarks/bench_ds
rm -rf dependencies/*
rm -rf outputs/*

echo "Clean complete. Run ./benchmark_pipeline.sh to rebuild."