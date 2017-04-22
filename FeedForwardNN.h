/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/


#ifndef FEEDFORWARDNN_H_
#define FEEDFORWARDNN_H_


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#include "ActivationFunctions.h"
#include "LearningSet.h"


#include "RandomGenerator.h"

#define INITWEIGHTMAX 0.1

class FeedForwardNN {
public:
    FeedForwardNN() {}

    // constructor with int (number of layers), array (layer sizes), array (activation functions)
    FeedForwardNN::FeedForwardNN(const int num, const int * siz, const int * funct) {
        if (num<2) { printf("BAD NETWORK INITIALIZATION\n"); exit(1); }

        numOfLayers = num;
        layersSize.resize(numOfLayers);
        for (int i = 0; i<numOfLayers; i++) {
            layersSize[i] = siz[i];
        }
        actFuncts.resize(numOfLayers);
        for (int i = 0; i<numOfLayers; i++) {
            actFuncts[i] = funct[i];
        }
        numOfWeights = 0;
        for (int i = 0; i<numOfLayers - 1; i++) {
            numOfWeights += (layersSize[i] + 1)*layersSize[i + 1];
        }
        weights.resize(numOfWeights);
        initWeights();
    }

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
    FeedForwardNN(const char * s) {
        FILE * f;
        f = fopen(s, "r");

        //file not found
        if (f != NULL) {
            //file wrong format
            if (fscanf(f, "%d", &numOfLayers)<1) { printf("WRONG NETWORK FILE FORMAT\n"); exit(1); }
            layersSize.resize(numOfLayers);
            for (int i = 0; i<numOfLayers; i++)
                if (fscanf(f, "%d", &layersSize[i])<1) { printf("WRONG NETWORK FILE FORMAT\n"); exit(1); }
            actFuncts.resize(numOfLayers);
            for (int i = 0; i<numOfLayers; i++)
                if (fscanf(f, "%d", &actFuncts[i])<1) { printf("WRONG NETWORK FILE FORMAT\n"); exit(1); }
            if (fscanf(f, "%d", &numOfWeights)<1) { printf("WRONG NETWORK FILE FORMAT\n"); exit(1); }
            weights.resize(numOfWeights);
            for (int i = 0; i<numOfWeights; i++)
                if (fscanf(f, "%f", &weights[i])<1) { printf("WRONG NETWORK FILE FORMAT\n"); exit(1); }
            fclose(f);
        }
        else { printf("COULDN'T OPEN THE NETWORK FILE\n"); exit(1); }
    }

    // initialize randomly the network weights between min and max
    void initWeights(float min = -INITWEIGHTMAX, float max = INITWEIGHTMAX) {
        for (int i = 0; i<numOfWeights; i++) {
            //TEST WEIGHTS
            //weights[i]=2;
            //weights[i]=(2*max*gen_random_real(0, 1))-max;
            weights[i] = gen_random_real(min, max);
        }
    }

    // initialize the network weights with Widrow Nguyen algorithm
    void initWidrowNguyen(LearningSet & set) {
        float min = set.getInputs()[0];
        float max = set.getInputs()[0];

        //finds the min and max value of inputs
        for (int i = 0; i<set.getNumOfInstances()*set.getNumOfInputsPerInstance(); i++) {
            float val = set.getInputs()[i];
            if (val<min)
                min = val;
            if (val>max)
                max = val;
        }

        int nOfHid = 0;
        for (int i = 1; i<numOfLayers - 1; i++)
            nOfHid += layersSize[i];
        float mult = (float)(pow((double)(0.7f*(double)nOfHid), (double)(1.0f / (double)layersSize[0])) / (double)(max - min));


        std::vector<int> offsetWeights(numOfLayers);
        for (int i = 0; i<numOfLayers; i++) {
            offsetWeights[i] = 0;
            for (int j = 0; j<i; j++) {
                offsetWeights[i] += (layersSize[j] + 1)*layersSize[j + 1];
            }
        }
        for (int i = 0; i<numOfLayers - 1; i++)
            for (int j = 0; j<layersSize[i + 1]; j++)
                for (int k = 0; k<layersSize[i] + 1; k++)
                    if (k<layersSize[i]) {
                        weights[offsetWeights[i] + j*(layersSize[i] + 1) + k] = mult * gen_random_real(0, 1);

                    }
                    else
                        weights[offsetWeights[i] + j*(layersSize[i] + 1) + k] = 2 * mult*gen_random_real(0, 1) - mult;
    }

    // computes the net outputs
    void FeedForwardNN::compute(const float * inputs, float * outputs) const {

        int offset = 0;

        std::vector<float> in(layersSize[0] + 1);
        //loads the inputs
        for (int i = 0; i<layersSize[0]; i++)
            in[i] = inputs[i];

        std::vector<float> out;

        //loops the layers
        for (int i = 0; i<numOfLayers - 1; i++) {

            //bias
            in[layersSize[i]] = 1.0;

            offset = 0;
            for (int j = 0; j<i; j++) {
                offset += (layersSize[j] + 1)*layersSize[j + 1];
            }

            out.resize(layersSize[i + 1]);

            float tot = 0;


            //loops the outputs
            for (int j = 0; j<layersSize[i + 1]; j++) {
                tot = 0;

                //loops the inputs
                for (int k = 0; k<layersSize[i] + 1; k++) {
                    tot += in[k] * weights[k + j*(layersSize[i] + 1) + offset];
                }
                out[j] = actFunction(actFuncts[i + 1], tot);
            }


            in.resize(layersSize[i + 1] + 1);
            for (int l = 0; l<layersSize[i + 1]; l++) {
                in[l] = out[l];
            }
        }

        for (int i = 0; i<layersSize[numOfLayers - 1]; i++)
            outputs[i] = out[i];

    }

    // computes the MSE on a set
    float computeMSE(LearningSet & set) const {
        float mse = 0;

        int numOfInstances = set.getNumOfInstances();
        int numOfInputsPerInstance = set.getNumOfInputsPerInstance();
        int numOfOutputsPerInstance = set.getNumOfOutputsPerInstance();

        std::vector<float> netOuts(numOfOutputsPerInstance);

        //local variables for faster access
        float * inputs = set.getInputs();
        float * outputs = set.getOutputs();

        for (int instance = 0; instance<numOfInstances; instance++) {
            //compute using the inputs with an offset to point to each instance
            compute(inputs + instance*numOfInputsPerInstance, &netOuts[0]);
            for (int i = 0; i<numOfOutputsPerInstance; i++) {
                float x = outputs[i + instance*numOfOutputsPerInstance] - netOuts[i];
                mse += x*x;
            }
        }

        mse /= (numOfInstances*numOfOutputsPerInstance)*spanSize(actFuncts[numOfLayers - 1])*spanSize(actFuncts[numOfLayers - 1]);
        return mse;
    }

    // returns the index of the most high output neuron (classification)
    int classificate(const float * inputs) const {
        int outputsSize = layersSize[numOfLayers - 1];
        float max = -10000;
        int indmax = 0;
        std::vector<float> outputs(outputsSize);

        compute(inputs, &outputs[0]);
        for (int j = 0; j<outputsSize; j++) {
            if (outputs[j]>max) {
                indmax = j;
                max = outputs[j];
            }
        }
        return indmax;
    }

    // computes the fraction of correct classification on a set (0 to 1)
    float classificatePerc(LearningSet & set) const {
        int cont = 0;
        int numOfInstances = set.getNumOfInstances();
        int numOfInputsPerInstance = set.getNumOfInputsPerInstance();
        int numOfOutputsPerInstance = set.getNumOfOutputsPerInstance();

        for (int i = 0; i<numOfInstances; i++) {
            if (set.getOutputs()[classificate(set.getInputs() + i*numOfInputsPerInstance) + i*numOfOutputsPerInstance] == 1) {
                cont++;
            }
        }
        return (float)cont / (float)numOfInstances;
    }

    // saves the network to a txt file
    void saveToTxt(const char * s) const {
        FILE * f;
        f = fopen(s, "w");
        fprintf(f, "%d\n", numOfLayers);
        for (int i = 0; i<numOfLayers; i++)
            fprintf(f, "%d ", layersSize[i]);
        fprintf(f, "\n");
        for (int i = 0; i<numOfLayers; i++)
            fprintf(f, "%d ", actFuncts[i]);
        fprintf(f, "\n%d\n", numOfWeights);
        for (int i = 0; i<numOfWeights; i++)
            fprintf(f, "%.20e\n", weights[i]);
        fclose(f);
    }

    const int * getLayersSize() const {
        return &layersSize[0];
    }

    int getNumOfLayers() const {
        return numOfLayers;
    }

    int getNumOfWeights() const {
        return numOfWeights;
    }

    float * getWeights() {
        return &weights[0];
    }



    const int * getActFuncts() const {
        return &actFuncts[0];
    }

    float getWeight(int ind) const {
        return weights[ind];
    }

    void setWeight(int ind, float weight) {
        weights[ind] = weight;
    }
    

private:
    int numOfLayers = 0;
    std::vector<int> layersSize;
    std::vector<int> actFuncts;
    int numOfWeights = 0;
    std::vector<float> weights;
};

#endif /* FEEDFORWARDNN_H_ */
