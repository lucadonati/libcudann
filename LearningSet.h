/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

#pragma once


#include <cassert>
#include <vector>
#include <fstream>
#include <sstream>

#include <random>
#include <algorithm>

#include <string>


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
    static LearningSet readFannSet(const char * s) {
        LearningSet set;
        std::ifstream ifs(s);
        //file not found
        if (!ifs)
            throw std::runtime_error(std::string("Couldn't open the Fann learning set file: ") + s);
        auto check_bad_format = [&]() {
            if (!ifs)
                throw std::runtime_error(std::string("Wrong Fann learning set file format: ") + s);
        };
        // check for wrong file format
        ifs >> set.numOfInstances;
        check_bad_format();
        ifs >> set.numOfInputsPerInstance;
        check_bad_format();
        ifs >> set.numOfOutputsPerInstance;
        check_bad_format();

        set.inputs.resize(set.numOfInstances*set.numOfInputsPerInstance);
        set.outputs.resize(set.numOfInstances*set.numOfOutputsPerInstance);
        for (int i = 0; i<set.numOfInstances; i++) {
            for (int j = 0; j < set.numOfInputsPerInstance; j++) {
                ifs >> set.inputs[i*set.numOfInputsPerInstance + j];
                check_bad_format();
            }
            for (int j = 0; j < set.numOfOutputsPerInstance; j++) {
                ifs >> set.outputs[i*set.numOfOutputsPerInstance + j];
                check_bad_format();
            }
        }
        return set;
    }
    void writeFannSet(const char * s) {
        std::ofstream ofs(s);
        ofs << numOfInstances << " " << numOfInputsPerInstance << " " << numOfOutputsPerInstance << "\n";

        write_instances_to_ofs(ofs);
    }
    /// constructor from txt file (simplified format)
    /// format is:
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
    /// spaces do not matter. each line must end with \n. empty lines are fine
    static LearningSet readSimplifiedSet(const char * s) {
        LearningSet set;
        std::ifstream ifs(s);
        //file not found
        if (!ifs)
            throw std::runtime_error(std::string("Couldn't open the Fann learning set file: ") + s);
        auto check_bad_format = [&]() {
            if (!ifs)
                throw std::runtime_error(std::string("Wrong Fann learning set file format: ") + s);
        };
        
        bool in = true;
        int totlines = 0;
        std::string line;
        while (std::getline(ifs, line)) {
            if (!ifs)
                break;
            std::istringstream iss(line);
            while (iss) {
                float tmp = 0.0;
                iss >> tmp;
                if (iss.fail())
                    break;
                if(in)
                    set.inputs.push_back(tmp);
                else
                    set.outputs.push_back(tmp);
            }
            in = !in;
            ++totlines;
        }

        
        set.numOfInstances = totlines / 2;

        if (totlines % 2 == 1)
            throw std::runtime_error("Odd number of lines");

        set.numOfInputsPerInstance = int(set.inputs.size()) / set.numOfInstances;
        if (int(set.inputs.size() % set.numOfInstances) != 0)
            throw std::runtime_error("Wrong number of inputs");

        set.numOfOutputsPerInstance = int(set.outputs.size()) / set.numOfInstances;
        if (int(set.outputs.size() % set.numOfInstances) != 0)
            throw std::runtime_error("Wrong number of outputs");
        
        return set;
    }
    void writeSimplifiedSet(const char * s) {
        std::ofstream ofs(s);

        write_instances_to_ofs(ofs);
    }

    static LearningSet readBinarySet(const char * s) {
        LearningSet set;
        std::ifstream ifs(s, std::ofstream::binary);

        auto bin_read = [] (auto && s, auto && t) {
            s.read(reinterpret_cast<char*>(&t), sizeof(t));
        };

        bin_read(ifs, set.numOfInstances);
        bin_read(ifs, set.numOfInputsPerInstance);
        bin_read(ifs, set.numOfOutputsPerInstance);

        set.inputs.resize(set.numOfInstances * set.numOfInputsPerInstance);
        set.outputs.resize(set.numOfInstances * set.numOfOutputsPerInstance);

        int i_ind = 0;
        int o_ind = 0;
        for (int n = 0; n < set.numOfInstances; ++n) {
            for (int i = 0; i < set.numOfInputsPerInstance; ++i)
                bin_read(ifs, set.inputs[i_ind++]);

            for (int i = 0; i < set.numOfOutputsPerInstance; ++i)
                bin_read(ifs, set.outputs[o_ind++]);
        }
        return set;
    }
    void writeBinarySet(const char * s) {
        std::ofstream ofs(s, std::ofstream::binary);

        auto bin_write = [](auto && s, auto && t) {
            s.write(reinterpret_cast<char*>(&t), sizeof(t));
        };

        bin_write(ofs, numOfInstances);
        bin_write(ofs, numOfInputsPerInstance);
        bin_write(ofs, numOfOutputsPerInstance);

        int i_ind = 0;
        int o_ind = 0;
        for (int n = 0; n < numOfInstances; ++n) {
            for (int i = 0; i < numOfInputsPerInstance; ++i)
                bin_write(ofs, inputs[i_ind++]);
            
            for (int i = 0; i < numOfOutputsPerInstance; ++i)
                bin_write(ofs, outputs[o_ind++]);
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
    
    void insert(const LearningSet & set) {
        if (numOfInputsPerInstance == 0)
            numOfInputsPerInstance = set.numOfInputsPerInstance;
        if (numOfOutputsPerInstance == 0)
            numOfOutputsPerInstance = set.numOfOutputsPerInstance;

        assert(numOfInputsPerInstance == set.numOfInputsPerInstance);
        assert(numOfOutputsPerInstance == set.numOfOutputsPerInstance);


        inputs.insert(inputs.end(), set.inputs.begin(), set.inputs.end());
        outputs.insert(outputs.end(), set.outputs.begin(), set.outputs.end());

        numOfInstances += set.numOfInstances;
    }

private:
    template<typename OFS>
    void write_instances_to_ofs(OFS && ofs) {
        int i_ind = 0;
        int o_ind = 0;
        for (int n = 0; n < numOfInstances; ++n) {

            for (int i = 0; i < numOfInputsPerInstance; ++i)
                ofs << inputs[i_ind++] << " ";
            ofs << "\n";

            for (int i = 0; i < numOfOutputsPerInstance; ++i)
                ofs << outputs[o_ind++] << " ";
            ofs << "\n";
        }
    }
    int numOfInstances = 0;
    int numOfInputsPerInstance = 0;
    int numOfOutputsPerInstance = 0;

    std::vector<float> inputs;
    std::vector<float> outputs;
};





