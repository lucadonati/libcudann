/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

#pragma once

const int ERROR_LINEAR = 0;
const int ERROR_TANH   = 1;

#ifdef __CUDACC__
__host__ __device__
#endif
//returns the new error after the application of a function (tanh is more aggressive error targeting)
inline float errorFunction(float error, int func){
    if (func == ERROR_TANH) {
        if (error < -.9999999)        return -17.0;
        else if (error > .9999999)       return 17.0;
        else                                 return log((1.0 + error) / (1.0 - error));
    }
    else
        return error;
}
