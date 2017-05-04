/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

#pragma once


#include <vector>

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
        FILE * f;
        f = fopen(s, "r");
        //file not found
        if (f != NULL) {
            //file wrong format
            if (fscanf(f, "%d", &numOfInstances)<1) { printf("WRONG LEARNING SET FILE FORMAT\n"); exit(1); }
            if (fscanf(f, "%d", &numOfInputsPerInstance)<1) { printf("WRONG LEARNING SET FILE FORMAT\n"); exit(1); }
            if (fscanf(f, "%d", &numOfOutputsPerInstance)<1) { printf("WRONG LEARNING SET FILE FORMAT\n"); exit(1); }
            inputs.resize(numOfInstances*numOfInputsPerInstance);
            outputs.resize(numOfInstances*numOfOutputsPerInstance);
            for (int i = 0; i<numOfInstances; i++) {
                for (int j = 0; j<numOfInputsPerInstance; j++)
                    if (fscanf(f, "%f", &inputs[i*numOfInputsPerInstance + j])<1) { printf("WRONG LEARNING SET FILE FORMAT\n"); exit(1); }
                for (int j = 0; j<numOfOutputsPerInstance; j++)
                    if (fscanf(f, "%f", &outputs[i*numOfOutputsPerInstance + j])<1) { printf("WRONG LEARNING SET FILE FORMAT\n"); exit(1); }
            }
            fclose(f);
        }
        else { printf("COULDN'T OPEN THE LEARNING SET FILE\n"); exit(1); }
    }
    
    auto getNumOfInputsPerInstance() const {
        return numOfInputsPerInstance;
    }
    auto getNumOfInstances() const {
        return numOfInstances;
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

private:
    int numOfInstances = 0;
    int numOfInputsPerInstance = 0;
    int numOfOutputsPerInstance = 0;

    std::vector<float> inputs;
    std::vector<float> outputs;
};
