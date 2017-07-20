/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

#pragma once

#include <stdexcept>

const int ACT_LINEAR  = 0;
const int ACT_SIGMOID = 1;
const int ACT_TANH    = 2;
const int ACT_RELU    = 3;


template <typename T1, typename T2, typename T3>
#ifdef __CUDACC__
__host__ __device__
#endif
inline T1 clip(T1 && x, T2 && lo, T3 && hi) {
    return (x < lo) ? lo : ((x > hi) ? (hi) : (x));
}

 /// returns the value of the activation function
inline float actFunction(int act, float x) {
    switch (act) {
        case ACT_LINEAR:    return x;
        case ACT_SIGMOID:   return 1.0f / (1.0f + exp(-x));
        case ACT_TANH:      return 2.0f / (1.0f + exp(-x)) - 1.0f;
        case ACT_RELU:      return x > 0 ? x : 0;
        default:            throw std::runtime_error("Function not yet implemented");
    }
}
/// returns the value of the activation function derivation (used for backpropagation)
inline float actDerivation(int act, float y) {
    switch (act) {
        case ACT_LINEAR:    return 1;
        case ACT_SIGMOID:   y = clip(y, 0.01f, 0.99f); return y*(1.0f - y);
        case ACT_TANH:      y = clip(y, -0.98f, 0.98f); return 0.5f*(1.0f - (y*y));
        case ACT_RELU:      return y > 0 ? 1 : 0;
        default:            throw std::runtime_error("Function not yet implemented");
    }
}
/// returns the span size of a function for error calculation
/// just used for last layer??
#ifdef __CUDACC__
__host__ __device__
#endif
inline float spanSize(int act) {
    if (act == ACT_TANH)
        return 2.0f;
    else
        return 1.0f;
}


