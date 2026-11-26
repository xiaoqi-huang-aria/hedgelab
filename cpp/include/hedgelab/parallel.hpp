#pragma once

// Deterministic multi-threading for Monte Carlo work.
//
// Work is split into fixed-size blocks of paths. Each block gets its own random
// generator seeded from (seed, block index), never one generator shared between
// threads, so there is no data race and results are bit-for-bit identical for any
// number of threads.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

namespace hedgelab {

inline int resolve_threads(int n_threads) {
    if (n_threads > 0) return n_threads;
    const unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : static_cast<int>(hw);
}

inline std::mt19937_64 block_rng(std::uint64_t seed, long long block) {
    const auto b = static_cast<std::uint64_t>(block);
    std::seed_seq seq{static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32),
                      static_cast<std::uint32_t>(b), static_cast<std::uint32_t>(b >> 32)};
    return std::mt19937_64(seq);
}

// Calls work(block, begin, end) for consecutive blocks of [0, n_items); threads take
// the next unprocessed block from a shared atomic counter. work must not throw.
template <class Work>
void parallel_blocks(long long n_items, long long block_size, int n_threads, Work&& work) {
    const long long n_blocks = (n_items + block_size - 1) / block_size;
    const long long workers = std::min<long long>(resolve_threads(n_threads), std::max(n_blocks, 1LL));
    std::atomic<long long> next{0};
    auto loop = [&] {
        for (long long b = next++; b < n_blocks; b = next++) {
            work(b, b * block_size, std::min(n_items, (b + 1) * block_size));
        }
    };
    std::vector<std::thread> pool;
    for (long long i = 1; i < workers; ++i) pool.emplace_back(loop);
    loop();
    for (auto& thread : pool) thread.join();
}

}  // namespace hedgelab
