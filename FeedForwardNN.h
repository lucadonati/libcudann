/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/


#ifndef FEEDFORWARDNN_H_
#define FEEDFORWARDNN_H_

#include <vector>

#include "ActivationFunctions.h"
#include "LearningSet.h"

#define INITWEIGHTMAX 0.1

class FeedForwardNN {
public:
    FeedForwardNN() {}

	// constructor with int (number of layers), array (layer sizes), array (activation functions)
	FeedForwardNN(const int, const int *, const int *);
	/* constructor from txt file
	 * format is:
	 *
	 * NUMBER_OF_LAYERS
	 * LAYER1_SIZE LAYER2_SIZE LAYER3_SIZE ...
	 * LAYER2_ACT_FUNC LAYER3_ACT_FUNC ...
	 * NUMBER_OF_WEIGHTS
	 * WEIGHT1
	 * WEIGHT2
	 * WEIGHT3
	 * .
	 * .
	 * .
	 *
	 * spaces or \n do not matter
	 */
	FeedForwardNN(const char *);

	// initialize randomly the network weights between min and max
	void initWeights(float min =-INITWEIGHTMAX,float max = INITWEIGHTMAX);
	// initialize the network weights with Widrow Nguyen algorithm
	void initWidrowNguyen(LearningSet &);
	// computes the net outputs
	void compute(const float *, float *) const;
	// computes the MSE on a set
	float computeMSE(LearningSet &) const;
	// returns the index of the most high output neuron (classification)
	int classificate(const float * inputs)  const;
	// computes the correct percentage of classification on a set (0 to 1)
	float classificatePerc(LearningSet &)  const;
	// saves the network to a txt file
	void saveToTxt(const char *) const;
	float getWeight(int ind) const;
    void setWeight(int ind, float weight);
    const int *getLayersSize() const;
    int getNumOfLayers() const;
    int getNumOfWeights() const;
    float * getWeights();
    const int * getActFuncts() const;
private:
	int numOfLayers = 0;
	std::vector<int> layersSize;
	std::vector<int> actFuncts;
	int numOfWeights = 0;
    std::vector<float> weights;
};

#endif /* FEEDFORWARDNN_H_ */
