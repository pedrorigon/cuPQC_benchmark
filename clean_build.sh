#!/bin/bash

echo "Cleaning build artifacts..."

rm -f benchmarks/bench_kem benchmarks/bench_ds
rm -rf dependencies/*
rm -rf outputs/*

echo "Clean complete. Run ./setup_benchmark.sh to rebuild."