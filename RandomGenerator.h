#pragma once

#include <random>

/// [from, to]
inline int gen_uniform_int(int from, int to) {
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(from, to);
    return dis(gen);
}

/// [from, to)
inline double gen_uniform_real(double from, double to) {
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(from, to);
    return dis(gen);
}

inline double gen_gaussian_real(double mean, double stddev) {
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    std::normal_distribution<> dis(mean, stddev);
    return dis(gen);
}