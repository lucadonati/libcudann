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
    FeedForwardNN(const std::vector<int> & siz, const std::vector<int> & funct) {
        if (siz.size()<2)
            throw std::runtime_error("Bad network initialization");
        if (siz.size() != funct.size())
            throw std::runtime_error("Sizes and functions of different dimensionality");

        numOfLayers = siz.size();
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
    void compute(const float * inputs, float * outputs,
                 std::vector<float> & buffer = std::vector<float>()) const {
        buffer.resize(getNumOfNeurons());
        //load an array of inputs
        for (int i = 0; i < getLayersSize()[0]; i++)
            buffer[i] = inputs[i];

        feedforward(&buffer[0]);
        auto out_offs = get_output_offsets();
        for (int i = 0; i<layersSize[numOfLayers - 1]; i++)
            outputs[i] = buffer[out_offs[numOfLayers - 2] + i];

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
    const auto * getWeights() const {
        return &weights[0];
    }

    const auto * getActFuncts() const {
        return &actFuncts[0];
    }
    /// this includes biases
    int getNumOfNeurons() const {
        int numOfNeurons = 0;
        for (int i = 0; i < getNumOfLayers(); i++) {
            numOfNeurons += getLayersSize()[i] + 1;
        }
        return numOfNeurons;
    }
    
    std::vector<int> get_weight_offsets(int scaleFactor = 1) const {
        std::vector<int> offsetWeights(numOfLayers);
        for (int i = 0; i < numOfLayers; i++) {
            offsetWeights[i] = 0;
            for (int j = 0; j < i; j++) {
                offsetWeights[i] += ((layersSize[j] + 1) * layersSize[j + 1]) * scaleFactor;
            }
        }
        return offsetWeights;
    }
    std::vector<int> get_input_offsets(int scaleFactor = 1) const {
        std::vector<int> offsetIns(numOfLayers);
        for (int i = 0; i < numOfLayers; i++) {
            offsetIns[i] = 0;
            for (int j = 0; j < i; j++) {
                offsetIns[i] += (layersSize[j] + 1) * scaleFactor;
            }
        }
        return offsetIns;
    }
    std::vector<int> get_output_offsets(int scaleFactor = 1) const {
        std::vector<int> offsetOuts(numOfLayers);
        for (int i = 0; i < numOfLayers; i++) {
            offsetOuts[i] = (layersSize[0] + 1) * scaleFactor;
            for (int j = 0; j < i; j++) {
                offsetOuts[i] += (layersSize[j + 1] + 1) * scaleFactor;
            }
        }
        return offsetOuts;
    }


    void feedforward(float * values) const {
        const auto offsetIns = get_input_offsets();
        const auto offsetWeights = get_weight_offsets();
        const auto offsetOuts = get_output_offsets();

        const auto * weights = getWeights();
        const auto * actFuncts = getActFuncts();
        int numOfLayers = getNumOfLayers();
        const auto * layersSize = getLayersSize();        


        

        //loops the layers
        for (int i = 0; i < numOfLayers - 1; i++) {

            //bias neuron
            values[offsetIns[i] + layersSize[i]] = 1.0;

            float tot = 0;
            //loops the outputs
            for (int j = 0; j<layersSize[i + 1]; j++) {
                //unrolled sum of all to avoid some floating points precision problems
                tot = 0;
                int k = (layersSize[i] + 1) % 4;
                int off = j*(layersSize[i] + 1) + offsetWeights[i];
                switch (k) {
                    case 3: tot += weights[off + 2] * values[2 + offsetIns[i]];
                    case 2: tot += weights[off + 1] * values[1 + offsetIns[i]];
                    case 1: tot += weights[off    ] * values[offsetIns[i]];
                    case 0:;
                }
                for (; k<layersSize[i] + 1; k += 4) {
                    tot += weights[off + k    ] * values[k + offsetIns[i]] +
                           weights[off + k + 1] * values[k + 1 + offsetIns[i]] +
                           weights[off + k + 2] * values[k + 2 + offsetIns[i]] +
                           weights[off + k + 3] * values[k + 3 + offsetIns[i]];
                }
                //write the ouputs of the layer
                values[j + offsetOuts[i]] = actFunction(actFuncts[i + 1], tot);
            }
        }
    }

private:
    int numOfLayers = 0;
    std::vector<int> layersSize;
    std::vector<int> actFuncts;
    int numOfWeights = 0;
    std::vector<float> weights;
};
