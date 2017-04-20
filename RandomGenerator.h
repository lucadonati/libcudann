#pragma once

#include <random>

/// [from, to]
inline int gen_random_int(int from, int to) {
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(from, to);
    return dis(gen);
}

/// [from, to)
inline double gen_random_real(double from, double to) {
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(from, to);
    return dis(gen);
}