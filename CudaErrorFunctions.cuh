/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

#pragma once

#include "CudaActivationFunctions.cuh"
#include "ErrorFunctions.h"

//this macro computes the new error after the application of a function (tanh is more aggressive error targeting)
#define calcErr(error,errorFunc)(\
    errorFunc == ERROR_TANH ?\
    error < -.9999999f ? -17.0f:error > .9999999f ? 17.0f: log((1.0f + error) / (1.0f - error)):\
    error\
)

//computes the error function for (number) elements of (desired)-(neurons) and store the results in (deltas)
void computeError(float * deltas, const float * desired, const float * neurons, const int number, const int actFunc, const int errorFunc);
//computes the total mse for (number) elements of (desired)-(neurons)
float mseError(const float * desired, float * neurons, const int number, const int actFunc);
//find the (indexes) of the max values of each row of a set of (neurons), divided in rows(nOfOut) and columns(nOfInst)
void computeMaxes(const int nOfInst, const int nOfOut, const float * neurons, int * indexes);
//adds to (number) elements of (weights) the difference between (weights) and (oldWeights) multiplied with (momentum)
void addMomentum(float * weights, float * oldWeights,const int number, const float momentum);
//translate a matrix width-height (rows large (width) and columns high (height)) to one height-width
void translateMatrix(const int width, const int height, const float * in, float * out);
