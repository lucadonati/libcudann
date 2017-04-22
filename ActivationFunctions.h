/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

/*
 * ActivationFunctions.h
 *
 *  Created on: Nov 24, 2010
 *      Author: donati
 */

#ifndef ACTIVATIONFUNCTIONS_H_
#define ACTIVATIONFUNCTIONS_H_

#define ACT_LINEAR        0
#define ACT_SIGMOID        1
#define ACT_TANH        2
#define ACT_RELU        3

#define clip(x, lo, hi) (((x) < (lo)) ? (lo) : (((x) > (hi)) ? (hi) : (x)))

 //returns the value of the activation function
inline float actFunction(int act, float x) {
    switch (act) {
        case ACT_LINEAR:    return x;//printf("LINEAR SHOULD NOT BE USED FOR NOW\n");exit(1);
        case ACT_SIGMOID:    return 1.0f / (1.0f + exp(-x));
        case ACT_TANH:        return 2.0f / (1.0f + exp(-x)) - 1.0f;
        case ACT_RELU:      return x > 0 ? x : 0;
        default:            printf("FUNCTION NOT IMPLEMENTED YET\n"); exit(1);
    }
}
//returns the value of the activation function derivation (used for backpropagation)
inline float actDerivation(int act, float y) {
    switch (act) {
        case ACT_LINEAR:    return 1;//printf("LINEAR SHOULD NOT BE USED FOR DERIVATION\n");exit(1);
        case ACT_SIGMOID:    y = clip(y, 0.01f, 0.99f); return y*(1.0f - y);
        case ACT_TANH:        y = clip(y, -0.98f, 0.98f); return 0.5f*(1.0f - (y*y));
        case ACT_RELU:      return y > 0 ? 1 : 0;
        default:            printf("FUNCTION NOT IMPLEMENTED YET\n"); exit(1);
    }
}
//returns the span size of a function for error calculation
// just used for last layer??
inline float spanSize(int act) {
    switch (act) {
        case ACT_LINEAR:
        case ACT_SIGMOID:    return 1.0f;
        case ACT_TANH:        return 2.0f;
        case ACT_RELU:      return 1.0f;
        default:            printf("FUNCTION NOT IMPLEMENTED YET\n"); exit(1);
    }
}

#endif /* ACTIVATIONFUNCTIONS_H_ */
