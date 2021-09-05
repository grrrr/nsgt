#!/usr/bin/env bash

echo "matrixform"

echo "torch 2070 super"
python examples/benchmark.py --torch-device=cuda:1 --bench-iter=100 --scale=bark --fmin=20 --fmax=22050 --bins=50 --real --matrixform ./mix.wav
echo "torch cpu"
python examples/benchmark.py --torch-device=cpu --bench-iter=100 --scale=bark --fmin=20 --fmax=22050 --bins=50 --real --matrixform ./mix.wav
echo "torch 3080 ti"
python examples/benchmark.py --torch-device=cuda:0 --bench-iter=100 --scale=bark --fmin=20 --fmax=22050 --bins=50 --real --matrixform ./mix.wav
echo "old nonmultithread"
python examples/benchmark.py --torch-device=cpu --bench-iter=100 --scale=bark --fmin=20 --fmax=22050 --bins=50 --real --matrixform --old ./mix.wav
echo "old multithread"
python examples/benchmark.py --torch-device=cpu --bench-iter=100 --scale=bark --fmin=20 --fmax=22050 --bins=50 --real --matrixform --old --multithreading ./mix.wav

echo "ragged"
echo "torch 2070 super"
python examples/benchmark.py --torch-device=cuda:1 --bench-iter=100 --scale=bark --fmin=20 --fmax=22050 --bins=50 --real ./mix.wav
echo "torch cpu"
python examples/benchmark.py --torch-device=cpu --bench-iter=100 --scale=bark --fmin=20 --fmax=22050 --bins=50 --real ./mix.wav
echo "torch 3080 ti"
python examples/benchmark.py --torch-device=cuda:0 --bench-iter=100 --scale=bark --fmin=20 --fmax=22050 --bins=50 --real ./mix.wav
echo "old nonmultithread"
python examples/benchmark.py --torch-device=cpu --bench-iter=100 --scale=bark --fmin=20 --fmax=22050 --bins=50 --real --old ./mix.wav
echo "old multithread"
python examples/benchmark.py --torch-device=cpu --bench-iter=100 --scale=bark --fmin=20 --fmax=22050 --bins=50 --real --old --multithreading ./mix.wav
