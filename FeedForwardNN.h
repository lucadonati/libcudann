/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

#pragma once


#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>

#include <vector>

#include "ActivationFunctions.h"
#include "LearningSet.h"

#include "RandomGenerator.h"

const double INITWEIGHTMAX = 0.1;

class FeedForwardNN {
public:
    FeedForwardNN() {}

    /// constructor with int (number of layers), array (layer sizes), array (activation functions)
    FeedForwardNN(const int num, const int * siz, const int * funct) {
        if (num<2)
            throw std::runtime_error("Bad network initialization");

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
        initWeightsUniform();
    }

    ///constructor from txt file
    /// format is:
    ///
    /// NUMBER_OF_LAYERS
    /// LAYER1_SIZE LAYER2_SIZE LAYER3_SIZE ...
    /// LAYER2_ACT_FUNC LAYER3_ACT_FUNC ...
    /// NUMBER_OF_WEIGHTS
    /// WEIGHT1
    /// WEIGHT2
    /// WEIGHT3
    /// .
    /// .
    /// .
    ///
    /// spaces or \n do not matter
    ///
    FeedForwardNN(const char * s) {
        std::ifstream ifs(s);
        if (!ifs)
            throw std::runtime_error(std::string("Couldn't open the network file: ") + s);
        auto check_bad_format = [&]() {
            if (!ifs)
                throw std::runtime_error(std::string("Wrong network file format: ") + s);
        };
        ifs >> numOfLayers;
        check_bad_format();
        layersSize.resize(numOfLayers);
        for (int i = 0; i < numOfLayers; i++) {
            ifs >> layersSize[i];
            check_bad_format();
        }
        actFuncts.resize(numOfLayers);
        for (int i = 0; i < numOfLayers; i++) {
            ifs >> actFuncts[i];
            check_bad_format();
        }
        ifs >> numOfWeights;
        check_bad_format();
        weights.resize(numOfWeights);
        for (int i = 0; i < numOfWeights; i++) {
            ifs >> weights[i];
            check_bad_format();
        }
    }

    /// initialize randomly the network weights between min and max
    void initWeightsUniform(float min = -INITWEIGHTMAX, float max = INITWEIGHTMAX) {
        for (int i = 0; i<numOfWeights; i++) {
            //TEST WEIGHTS
            //weights[i]=2;
            //weights[i]=(2*max*gen_random_real(0, 1))-max;
            weights[i] = gen_uniform_real(min, max);
        }
    }
    void initWeightsGaussian(float stdev, float multiplier) {
        for (int i = 0; i<numOfWeights; i++) {
            weights[i] = gen_gaussian_real(0, stdev) * multiplier;
        }
    }

    /// initialize the network weights with Widrow Nguyen algorithm
    void initWeightsWidrowNguyen(const LearningSet & set) {
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
                        weights[offsetWeights[i] + j*(layersSize[i] + 1) + k] = mult * gen_uniform_real(0, 1);

                    }
                    else
                        weights[offsetWeights[i] + j*(layersSize[i] + 1) + k] = 2 * mult*gen_uniform_real(0, 1) - mult;
    }

    void initWeightsBengio(double scale = 1) {
        auto offs = get_weight_offsets();
        auto nlayers = getNumOfLayers();

        for (int i = 0; i < nlayers - 1; ++i) {
            auto from = &weights[0] + offs[i];
            auto to = &weights[0] + offs[i + 1];

            while (from != to) {
                //std::cout << std::sqrt(getLayersSize()[i]) << "\n";
                double r = std::sqrt(1.0 / (getLayersSize()[i] +
                                            getLayersSize()[i + 1]) * scale);
                *from = gen_uniform_real(-r, r);
                ++from;
            }
        }
    }



    /// computes the net outputs
    void compute(const float * inputs, float * outputs) const {

        

        std::vector<float> in(layersSize[0] + 1);
        //loads the inputs
        for (int i = 0; i<layersSize[0]; i++)
            in[i] = inputs[i];

        std::vector<float> out;

        //loops the layers
        for (int i = 0; i<numOfLayers - 1; i++) {

            //bias
            in[layersSize[i]] = 1.0;

            int offset = 0;
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

    /// computes the MSE on a set
    float computeMSE(const LearningSet & set) const {
        float mse = 0;

        int numOfInstances = set.getNumOfInstances();
        int numOfInputsPerInstance = set.getNumOfInputsPerInstance();
        int numOfOutputsPerInstance = set.getNumOfOutputsPerInstance();

        std::vector<float> netOuts(numOfOutputsPerInstance);

        //local variables for faster access
        auto * inputs = set.getInputs();
        auto * outputs = set.getOutputs();

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

    /// returns the index of the most high output neuron (classification)
    int classificate(const float * inputs) const {
        int outputsSize = layersSize[numOfLayers - 1];
        float max = std::numeric_limits<float>::min();
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

    /// computes the fraction of correct classification on a set (0 to 1)
    float classificatePerc(const LearningSet & set) const {
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

    /// saves the network to a txt file
    void saveToTxt(const char * s) const {
        std::ofstream ofs(s);
        if (!ofs)
            throw std::runtime_error(std::string("Failed to open file ") + s + " for saving network.");
        ofs << numOfLayers << "\n";
        for (int i = 0; i<numOfLayers; i++)
            ofs << layersSize[i];
        ofs << "\n";

        for (int i = 0; i<numOfLayers; i++)
            ofs << actFuncts[i];
        ofs << "\n";
        ofs << numOfWeights << "\n";

        ofs.precision(20);
        ofs << std::scientific;
        for (int i = 0; i<numOfWeights; i++)
            ofs << weights[i];
        if (!ofs)
            throw std::runtime_error(std::string("Error occourred while saving network ") + s);
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

    auto * getWeights() {
        return &weights[0];
    }

    const auto * getActFuncts() const {
        return &actFuncts[0];
    }
    
    std::vector<int> get_weight_offsets(int scaleFactor = 1) {
        std::vector<int> offsetWeights(numOfLayers);
        for (int i = 0; i < numOfLayers; i++) {
            offsetWeights[i] = 0;
            for (int j = 0; j < i; j++) {
                offsetWeights[i] += ((layersSize[j] + 1) * layersSize[j + 1]) * scaleFactor;
            }
        }
        return offsetWeights;
    }
    std::vector<int> get_input_offsets(int scaleFactor = 1) {
        std::vector<int> offsetIns(numOfLayers);
        for (int i = 0; i < numOfLayers; i++) {
            offsetIns[i] = 0;
            for (int j = 0; j < i; j++) {
                offsetIns[i] += (layersSize[j] + 1) * scaleFactor;
            }
        }
        return offsetIns;
    }
    std::vector<int> get_output_offsets(int scaleFactor = 1) {
        std::vector<int> offsetOuts(numOfLayers);
        for (int i = 0; i < numOfLayers; i++) {
            offsetOuts[i] = (layersSize[0] + 1) * scaleFactor;
            for (int j = 0; j < i; j++) {
                offsetOuts[i] += (layersSize[j + 1] + 1) * scaleFactor;
            }
        }
        return offsetOuts;
    }

private:
    int numOfLayers = 0;
    std::vector<int> layersSize;
    std::vector<int> actFuncts;
    int numOfWeights = 0;
    std::vector<float> weights;
};
