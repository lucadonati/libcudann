/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

#pragma once


#include <vector>
#include <fstream>
#include <random>
#include <algorithm>


class LearningSet {
public:

    LearningSet() {}

     /// constructor from txt file (.fann format)
     /// format is:
     ///
     /// NUMBER_OF_ISTANCES
     /// NUMBER_OF_INPUTS_PER_ISTANCE
     /// NUMBER_OF_OUTPUTS_PER_ISTANCE
     ///
     /// INPUT1 INPUT2 INPUT3 ...
     /// OUTPUT1 OUTPUT2 OUTPUT3 ...
     ///
     /// INPUT1 INPUT2 INPUT3 ...
     /// OUTPUT1 OUTPUT2 OUTPUT3 ...
     ///
     /// INPUT1 INPUT2 INPUT3 ...
     /// OUTPUT1 OUTPUT2 OUTPUT3 ...
     ///
     /// .
     /// .
     /// .
     ///
     /// spaces or \n do not matter
    LearningSet(const char * s) {
        std::ifstream ifs(s);
        //file not found
        if (!ifs)
            throw std::runtime_error(std::string("Couldn't open the learning set file: ") + s);
        auto check_bad_format = [&]() {
            if (!ifs)
                throw std::runtime_error(std::string("Wrong learning set file format: ") + s);
        };
        // check for wrong file format
        ifs >> numOfInstances;
        check_bad_format();
        ifs >> numOfInputsPerInstance;
        check_bad_format();
        ifs >> numOfOutputsPerInstance;
        check_bad_format();

        inputs.resize(numOfInstances*numOfInputsPerInstance);
        outputs.resize(numOfInstances*numOfOutputsPerInstance);
        for (int i = 0; i<numOfInstances; i++) {
            for (int j = 0; j < numOfInputsPerInstance; j++) {
                ifs >> inputs[i*numOfInputsPerInstance + j];
                check_bad_format();
            }
            for (int j = 0; j < numOfOutputsPerInstance; j++) {
                ifs >> outputs[i*numOfOutputsPerInstance + j];
                check_bad_format();
            }
        }
    }
 
    auto getNumOfInstances() const {
        return numOfInstances;
    }
    auto getNumOfInputsPerInstance() const {
        return numOfInputsPerInstance;
    }
    auto getNumOfOutputsPerInstance() const {
        return numOfOutputsPerInstance;
    }

    auto * getInputs() {
        return &inputs[0];
    }
    const auto * getInputs() const {
        return &inputs[0];
    }
    auto * getOutputs() {
        return &outputs[0];
    }
    const auto * getOutputs() const {
        return &outputs[0];
    }

    const auto * get_input_n(int n) const {
        return &inputs[0] + n * numOfInputsPerInstance;
    }
    const auto * get_output_n(int n) const {
        return &outputs[0] + n * numOfOutputsPerInstance;
    }

    template <typename P>
    LearningSet filter_set_by(P predicate) {
        LearningSet filtered;
        filtered.numOfInputsPerInstance = getNumOfInputsPerInstance();
        filtered.numOfOutputsPerInstance = getNumOfOutputsPerInstance();
        for (int i = 0; i < numOfInstances; ++i) {
            if (predicate(i, get_input_n(i), get_output_n(i))) {
                filtered.inputs.insert(filtered.inputs.end(), get_input_n(i), get_input_n(i) + filtered.numOfInputsPerInstance);
                filtered.outputs.insert(filtered.outputs.end(), get_output_n(i), get_output_n(i) + filtered.numOfOutputsPerInstance);
                ++filtered.numOfInstances;
            }
        }
        return filtered;
    }

    LearningSet shuffle() {
        LearningSet sh;
        sh.numOfInstances = numOfInstances;
        sh.numOfInputsPerInstance = numOfInputsPerInstance;
        sh.numOfOutputsPerInstance = numOfOutputsPerInstance;
        

        std::vector<int> indexes;
        for (int i = 0; i < numOfInstances; ++i)
            indexes.emplace_back(i);

        thread_local std::random_device rd;
        thread_local std::mt19937 g(rd());
        std::shuffle(std::begin(indexes), std::end(indexes), g);

        for (int i = 0; i < numOfInstances; ++i) {
            auto off_in = get_input_n(indexes[i]);
            auto off_out = get_output_n(indexes[i]);
            sh.inputs.insert(sh.inputs.end(), off_in, off_in + numOfInputsPerInstance);
            sh.outputs.insert(sh.outputs.end(), off_out, off_out + numOfOutputsPerInstance);
        }
        return sh;
    }

private:
    int numOfInstances = 0;
    int numOfInputsPerInstance = 0;
    int numOfOutputsPerInstance = 0;

    std::vector<float> inputs;
    std::vector<float> outputs;
};





