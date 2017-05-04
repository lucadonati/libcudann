/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

#pragma once

#include "ActivationFunctions.h"

//computes the activation function for (number) elements of (neurons) and store the results in (neurons)
void computeActFunct(float * neurons, const int number, const int funct);

//computes the derivation function for (number) elements of (neurons) and multiplies and stores the results with and in (delta)
void computeDerivFunct(float * deltas, const float * neurons, const int number, const int funct);
