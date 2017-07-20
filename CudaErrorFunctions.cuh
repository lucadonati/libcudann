/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

#pragma once

#include "CudaActivationFunctions.cuh"
#include "ErrorFunctions.h"

//computes the error function for (number) elements of (desired)-(neurons) and store the results in (deltas)
void computeError(const float * desired, const float * neurons, float * deltas, int number, int actFunc, int errorFunc);
//computes the total mse for (number) elements of (desired)-(neurons)
float mseError(const float * desired, float * neurons, int number, int actFunc);
//find the (indexes) of the max values of each row of a set of (neurons), divided in rows(nOfOut) and columns(nOfInst)
void computeMaxes(int nOfInst, int nOfOut, const float * neurons, int * indexes);
//adds to (number) elements of (weights) the difference between (weights) and (oldWeights) multiplied with (momentum)
void addMomentum(float * weights, float * oldWeights, int number, float momentum);
//translate a matrix width-height (rows large (width) and columns high (height)) to one height-width
void translateMatrix(int width, int height, const float * in, float * out);
