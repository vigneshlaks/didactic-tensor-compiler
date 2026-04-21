#include "../include/autotuner.h"
#include <iostream>

static const int CANDIDATES[] = {8, 16, 32};

// Map from key (defined below) to integer
static std::unordered_map<uint64_t, int> g_tileCache;

uint64_t shapeKey(int M, int N, int K) {
    // Shift by 40 or 20 or 0
    return ((uint64_t)(uint32_t)M << 40) |
           ((uint64_t)(uint32_t)N << 20) |
            (uint64_t)(uint32_t)K;
}

// Check if the best tile for this shape is found
bool lookupTile(int M, int N, int K, int& tileOut) {
    auto it = g_tileCache.find(shapeKey(M, N, K));
    if (it == g_tileCache.end()) return false;
    tileOut = it->second;
    return true;
}

void storeTile(int M, int N, int K, int tile) {
    g_tileCache[shapeKey(M, N, K)] = tile;
}

// 
AutotuneResult autotune(float* A, float* B, float* C,
                        int M, int N, int K,
                        BenchmarkFn benchmark) {
    std::cout << "[Autotune] Tuning matmul (" << M << "x" << N << "x" << K << ")\n";

    AutotuneResult best{CANDIDATES[0], 1e18};

    for (int tile : CANDIDATES) {
        try {
            // One warmup run, then one timed run
            benchmark(A, B, C, M, N, K, tile);
            double t = benchmark(A, B, C, M, N, K, tile);
            std::cout << "[Autotune]   tile=" << tile << " -> " << t * 1e6 << " us\n";
            if (t < best.bestTime) {
                best.bestTime = t;
                best.bestTile = tile;
            }
        } catch (...) {
            // Tile size may be invalid for this shape — skip it
        }
    }

    std::cout << "[Autotune]   winner: tile=" << best.bestTile << "\n";
    storeTile(M, N, K, best.bestTile);
    return best;
}
