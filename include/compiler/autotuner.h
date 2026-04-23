#ifndef AUTOTUNER_H
#define AUTOTUNER_H

#include <functional>
#include <unordered_map>
#include <cstdint>

// A backend provides this — run a matmul with the given tile size, return seconds elapsed
using BenchmarkFn = std::function<double(float* A, float* B, float* C,
                                         int M, int N, int K, int tile)>;

struct AutotuneResult {
    int   bestTile;
    double bestTime;
};

// Packs (M, N, K) into a single key for the cache
uint64_t shapeKey(int M, int N, int K);

// Benchmarks candidate tile sizes and returns the winner
AutotuneResult autotune(float* A, float* B, float* C,
                        int M, int N, int K,
                        BenchmarkFn benchmark);

// Global cache: shape → best tile size found so far
// Backends call lookupTile() before dispatching to avoid re-tuning
bool     lookupTile(int M, int N, int K, int& tileOut);
void     storeTile(int M, int N, int K, int tile);

#endif
