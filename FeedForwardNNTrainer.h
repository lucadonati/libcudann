/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

#pragma once


#include <cstdio>
#include <cstdlib>
//#include <csignal>
#include <vector>

#include "RandomGenerator.h"

#include "FeedForwardNN.h"
#include "LearningSet.h"
#include "ErrorFunctions.h"

#ifndef DISABLE_CUDA_NN
#include "CudaActivationFunctions.cuh"
#include "CudaErrorFunctions.cuh"

#include <cublas.h>
//#include <cutil_inline.h>
#endif


const int TRAIN_CPU = 0;
#ifndef DISABLE_CUDA_NN
const int TRAIN_GPU = 1;
#endif
const int ALG_BP    = 0;
const int ALG_BATCH = 1;
const int SHUFFLE_OFF = 0;
const int SHUFFLE_ON  = 1;

const int PRINT_ALL = 0;
const int PRINT_MIN = 1;
const int PRINT_OFF = 2;


struct TrainingParameters {
    int training_location = TRAIN_CPU;
    int training_algorithm = ALG_BATCH;

    float desired_error = 0.0;
    int max_epochs = 10000;
    int epochs_between_reports = 100;
    float learningRate = 0.001;
    float momentum = 0.7;

    int shuff = SHUFFLE_ON;
    int errorFunc = ERROR_LINEAR;
};

inline void print_parameters(const TrainingParameters & params) {
    std::cout << "Training on:\t\t";
    if(params.training_location == TRAIN_CPU)
        std::cout << "CPU\n";
    else
        std::cout << "GPU\n";
    std::cout << "Algorithm:\t\t";
    if (params.training_location == ALG_BP)
        std::cout << "Backpropagation\n";
    else
        std::cout << "Batch\n";
    std::cout << "Desired Error:\t\t" << params.desired_error << "\n";
    std::cout << "Max epochs:\t\t" << params.max_epochs << "\n";
    std::cout << "Epochs between reports:\t" << params.epochs_between_reports << "\n";
    std::cout << "Learning rate:\t\t" << params.learningRate << "\n";
    std::cout << "Momentum:\t\t" << params.momentum << "\n";
    if (params.shuff == SHUFFLE_ON)
        std::cout << "Shuffle:\t\tON\n";
    else
        std::cout << "Shuffle:\t\tOFF\n";
    if (params.errorFunc == ERROR_TANH)
        std::cout << "Error function:\t\tTANH\n";
    else
        std::cout << "Error function:\t\tLINEAR\n";
    std::cout << "\n";
}

class FeedForwardNNTrainer {
public:

    ///SIGINT handler
    bool quit = false;
    void terminate(int) {
        quit = true;
    }


    ///choose a net to operate on and save after the training
    void selectNet(FeedForwardNN & n) {
        net = &n;
    }
    ///choose the training set
    void selectTrainingSet(LearningSet & s) {
        trainingSet = &s;
    }
    ///choose the test set. if this is set the error rate is computed on test set instead of training set
    void selectTestSet(LearningSet & s) {
        testSet = &s;
    }
    ///choose a net to save the best network trained so far after each epoch. mse on test set is the criterion
    void selectBestMSETestNet(FeedForwardNN & n) {
        bestMSETestNet = &n;
    }
    ///choose a net to save the best network trained so far after each epoch. mse on train set + mse on test set is the criterion
    void selectBestMSETrainTestNet(FeedForwardNN & n) {
        bestMSETrainTestNet = &n;
    }
    ///choose a net to save the best network trained so far after each epoch. percentage as classifier is the criterion
    void selectBestClassTestNet(FeedForwardNN & n) {
        bestClassTestNet = &n;
    }

    ///starts the training using params. n is the number of parameters
    ///the first 2 elements of params are where the training will be executed (TRAIN_CPU,TRAIN_GPU)
    ///and the training algorithm (ALG_BP,ALG_BATCH...). the other parameters are algorithm dependent
    ///returns the best MSE on test set (or train set if test set isn't specified)
    ///printtype specifies how much verbose will be the execution (PRINT_ALL,PRINT_MIN,PRINT_OFF)
    float train(const TrainingParameters & params, const int printtype = PRINT_ALL) {
        //checks CTRL-C to interrupt training manually
        quit = false;
       // signal(SIGINT, terminate);

        setvbuf(stdout, (char*)NULL, _IONBF, 0);

        // checks for network and training set correct initialization
        if (!net)
            throw std::runtime_error("NEURAL NETWORK NOT SELECTED");
        if (!trainingSet)
            throw std::runtime_error("TRAINING SET NOT SELECTED");
        if ((trainingSet->getNumOfInputsPerInstance() != net->getLayersSize()[0])
            || (trainingSet->getNumOfOutputsPerInstance() != net->getLayersSize()[net->getNumOfLayers() - 1])) {
            throw std::runtime_error("NETWORK AND TRAINING SET OF DIFFERENT SIZE");
        }
        if (testSet &&
            (trainingSet->getNumOfInputsPerInstance() != testSet->getNumOfInputsPerInstance()
                || trainingSet->getNumOfOutputsPerInstance() != testSet->getNumOfOutputsPerInstance())) {
            throw std::runtime_error("TEST SET OF DIFFERENT SIZE");
        }

        if (printtype != PRINT_OFF) {
            std::cout << "Network:\t\t";
            std::cout << net->getLayersSize()[0];
            for (int i = 1; i<net->getNumOfLayers(); i++)
                std::cout << "x" << net->getLayersSize()[i];
            std::cout << "\n";
            std::cout << "Activation functions:\t";
            for (int i = 0; i<net->getNumOfLayers(); i++)
                std::cout << net->getActFuncts()[i] << " ";
            std::cout << "\n";
        }

        //select the right algorithm to execute training
        switch (params.training_location) {
            case TRAIN_CPU:
                switch (params.training_algorithm) {
                    case ALG_BP:        return trainCpuBp(params, printtype);                      break;
                    case ALG_BATCH:     return trainCpuBatch(params, printtype);                   break;
                    default:            throw std::runtime_error("TRAINING NOT IMPLEMENTED YET");  break;
                }
                break;
#ifndef DISABLE_CUDA_NN
            case TRAIN_GPU:
                switch (params.training_algorithm) {
                    case ALG_BP:        throw std::runtime_error("TRAINING NOT IMPLEMENTED YET");  break;
                    case ALG_BATCH:     return trainGPUBatch(params, printtype);                   break;
                    default:            throw std::runtime_error("TRAINING NOT IMPLEMENTED YET");  break;
                }
                break;
#endif
            default: throw std::runtime_error("TRAINING NOT IMPLEMENTED YET");
        }

        //stops checking CTRL-C
        //signal(SIGINT, SIG_DFL);
}

private:
    ///backpropagation training on host
    ///n is the number of parameters. parameters are (float array):
    ///desired error, max_epochs, epochs_between_reports, learning_rate, momentum (using momentum is 20% slower), shuffle (SHUFFLE_ON or SHUFFLE_OFF), error function (ERROR_TANH or ERROR_LINEAR)
    float trainCpuBp(const TrainingParameters & params, const int printtype) {

        if (printtype != PRINT_OFF)
            print_parameters(params);

        float mseTrain = FLT_MAX, mseTest = FLT_MAX;
        //declare some error values for evaluating the trained network and storing best results
        //best net MSE on test, best net MSE on train+test, best net as classifier on test
        float bestMSETest = FLT_MAX, bestMSETrainTest = FLT_MAX, bestClassTest = 0;


        //declare some network values
        int numOfLayers = net->getNumOfLayers();
        int numOfWeights = net->getNumOfWeights();
        int numOfNeurons = 0;
        for (int i = 0; i < net->getNumOfLayers(); i++) {
            numOfNeurons += net->getLayersSize()[i] + 1;
        }

        //declare an array of neurons to represent the neuron values
        std::vector<float> values(numOfNeurons);
        //declare an array of deltas to represent the gradients for the weight updates
        std::vector<float> deltas(numOfNeurons);
        //declare an array of weights to use for momentum
        std::vector<float> oldWeights(numOfWeights);
        //declare an array of temporary weights to use for batch and similar methods
        std::vector<float> tmpWeights(numOfWeights);


        //declare a pointer to the net weights
        float * weights = net->getWeights();
        //declare a pointer to the net activation functions
        const int * actFuncts = net->getActFuncts();
        //declare a pointer to the net layers size
        const int * layersSize = net->getLayersSize();

        //retrieve some offsets to manage array indexes of each layer 'i'
        auto offsetWeights = net->get_weight_offsets();
        auto offsetIns = net->get_input_offsets();
        auto offsetOuts = net->get_output_offsets();
        auto offsetDeltas = net->get_output_offsets();

        //save previous weights to use in momentum calculation
        for (int w = 0; w < numOfWeights; w++)
            oldWeights[w] = weights[w];


        //declare some training set values
        int numOfInstances = trainingSet->getNumOfInstances();
        int numOfInputsPerInstance = trainingSet->getNumOfInputsPerInstance();
        int numOfOutputsPerInstance = trainingSet->getNumOfOutputsPerInstance();

        //declare a pointer to the training set inputs
        float * trainingSetInputs = trainingSet->getInputs();
        //declare a pointer to the training set outputs
        float * trainingSetOutputs = trainingSet->getOutputs();


        //vector to shuffle training set
        std::vector<int> order(numOfInstances);
        for (int i = 0; i < numOfInstances; i++)
            order[i] = i;

        if (printtype == PRINT_ALL) {
            //compute starting error rates
            printf("Starting:\tError on train set %.10f", net->computeMSE(*trainingSet));
            if (testSet) {
                printf("\t\tError on test set %.10f", net->computeMSE(*testSet));
            }
            printf("\n");
        }

        //epochs training
        for (int epoch = 1; epoch <= params.max_epochs && quit == false; epoch++) {

            //shuffle instances
            int ind = 0, aux = 0;
            if (params.shuff == SHUFFLE_ON)
                for (int i = 0; i < numOfInstances; i++) {
                    ind = gen_random_int(i, numOfInstances - 1);
                    aux = order[ind];
                    order[ind] = order[i];
                    order[i] = aux;
                }

            //instances training
            for (int instance = 0; instance < numOfInstances; instance++) {

                //computes a single instance forward of the backpropagation training
                stepForward(&values[0], weights, actFuncts, numOfLayers, layersSize, numOfInputsPerInstance, trainingSetInputs, &offsetIns[0], &offsetWeights[0], &offsetOuts[0], &order[0], instance);

                //computes a single instance backward of the backpropagation training
                stepBack(&values[0], weights, &deltas[0], actFuncts, numOfLayers, layersSize, numOfOutputsPerInstance, trainingSetOutputs, &offsetWeights[0], &offsetDeltas[0], &offsetOuts[0], &order[0], instance, params.errorFunc);

                //update the weights using the deltas
                weightsUpdate(&values[0], weights, weights, &deltas[0], numOfLayers, layersSize, &offsetIns[0], &offsetWeights[0], &offsetDeltas[0], params.momentum, &oldWeights[0], params.learningRate);

            }

            if (params.epochs_between_reports > 0 && epoch % params.epochs_between_reports == 0) {

                mseTrain = net->computeMSE(*trainingSet);
                if (printtype == PRINT_ALL)
                    printf("Epoch\t%d\tError on train set %.10f", epoch, mseTrain);

                if (testSet) {

                    mseTest = net->computeMSE(*testSet);
                    if (mseTest<bestMSETest) {
                        bestMSETest = mseTest;
                        if (bestMSETestNet) {
                            *bestMSETestNet = *net;
                        }
                    }
                    if ((mseTrain + mseTest) < bestMSETrainTest && bestMSETrainTestNet) {
                        *bestMSETrainTestNet = *net;
                        bestMSETrainTest = mseTrain + mseTest;
                    }
                    if (printtype == PRINT_ALL)
                        printf("\t\tError on test set %.10f", mseTest);

                    if (bestClassTestNet) {
                        float per = net->classificatePerc(*testSet);
                        if (printtype == PRINT_ALL)
                            printf("\t\tClassification percentage on test set: %.1f%%", per * 100);
                        if (per>bestClassTest) {
                            *bestClassTestNet = *net;
                            bestClassTest = per;
                            if (printtype == PRINT_ALL)
                                printf(" ***");
                        }
                    }

                    if (mseTest <= params.desired_error) {
                        if (printtype == PRINT_ALL)
                            printf("\nDesired error reached on test set.\n");
                        break;
                    }

                }

                if (printtype == PRINT_ALL)
                    printf("\n");

                if (mseTrain <= params.desired_error && !testSet) {
                    if (printtype == PRINT_ALL)
                        printf("Desired error reached on training set.\n");
                    break;
                }
            }
        }

        if (printtype == PRINT_ALL)
            std::cout << "Training complete.\n";
        if (testSet)
            return bestMSETest;
        else
            return mseTrain;

    }

    ///batch training on host
    ///n is the number of parameters. parameters are (float array):
    ///desired error, max_epochs, epochs_between_reports, learning_rate, momentum (using momentum is 20% slower), shuffle (SHUFFLE_ON or SHUFFLE_OFF), error function (ERROR_TANH or ERROR_LINEAR)void FeedForwardNNTrainer::trainCpuBatch(const int n, const float * params){
    float trainCpuBatch(const TrainingParameters & params, const int printtype) {
        
        if (printtype != PRINT_OFF)
            print_parameters(params);

        float mseTrain = FLT_MAX, mseTest = FLT_MAX;
        //declare some error values for evaluating the trained network and storing best results
        //best net MSE on test, best net MSE on train+test, best net as classifier on test
        float bestMSETest = FLT_MAX, bestMSETrainTest = FLT_MAX, bestClassTest = 0;


        //declare some network values
        int numOfLayers = net->getNumOfLayers();
        int numOfWeights = net->getNumOfWeights();
        int numOfNeurons = 0;
        for (int i = 0; i<net->getNumOfLayers(); i++) {
            numOfNeurons += net->getLayersSize()[i] + 1;
        }

        //declare an array of neurons to represent the neuron values
        std::vector<float> values(numOfNeurons);
        //declare an array of deltas to represent the gradients for the weight updates
        std::vector<float> deltas(numOfNeurons);
        //declare an array of weights to use for momentum
        std::vector<float> oldWeights(numOfWeights);
        //declare an array of temporary weights to use for batch and similar methods
        std::vector<float> tmpWeights(numOfWeights);

        //declare a pointer to the net weights
        float * weights = net->getWeights();
        //declare a pointer to the net activation functions
        const int * actFuncts = net->getActFuncts();
        //declare a pointer to the net layers size
        const int * layersSize = net->getLayersSize();

        //retrieve some offsets to manage array indexes of each layer 'i'
        auto offsetWeights = net->get_weight_offsets();
        auto offsetIns = net->get_input_offsets();
        auto offsetOuts = net->get_output_offsets();
        auto offsetDeltas = net->get_output_offsets();


        //save previous weights to use in momentum calculation
        for (int w = 0; w<numOfWeights; w++)
            oldWeights[w] = weights[w];
        //resets temporary weights for batch
        for (int w = 0; w<numOfWeights; w++)
            tmpWeights[w] = 0;

        //declare some training set values
        int numOfInstances = trainingSet->getNumOfInstances();
        int numOfInputsPerInstance = trainingSet->getNumOfInputsPerInstance();
        int numOfOutputsPerInstance = trainingSet->getNumOfOutputsPerInstance();

        //declare a pointer to the training set inputs
        float * trainingSetInputs = trainingSet->getInputs();
        //declare a pointer to the training set outputs
        float * trainingSetOutputs = trainingSet->getOutputs();


        //vector to shuffle training set
        std::vector<int> order(numOfInstances);
        for (int i = 0; i<numOfInstances; i++)
            order[i] = i;

        if (printtype == PRINT_ALL) {
            //compute starting error rates
            printf("Starting:\tError on train set %.10f", net->computeMSE(*trainingSet));
            if (testSet) {
                printf("\t\tError on test set %.10f", net->computeMSE(*testSet));
            }
            printf("\n");
        }

        //epochs training
        for (int epoch = 1; epoch <= params.max_epochs && quit == false; epoch++) {

            //shuffle instances
            int ind = 0, aux = 0;
            if (params.shuff == SHUFFLE_ON)
                for (int i = 0; i<numOfInstances; i++) {
                    ind = gen_random_int(i, numOfInstances - 1);
                    aux = order[ind];
                    order[ind] = order[i];
                    order[i] = aux;
                }


            //instances training
            for (int instance = 0; instance<numOfInstances; instance++) {

                //computes a single instance forward of the backpropagation training
                stepForward(&values[0], weights, actFuncts, numOfLayers, layersSize, numOfInputsPerInstance, trainingSetInputs, &offsetIns[0], &offsetWeights[0], &offsetOuts[0], &order[0], instance);

                //computes a single instance backward of the backpropagation training
                stepBack(&values[0], weights, &deltas[0], actFuncts, numOfLayers, layersSize, numOfOutputsPerInstance, trainingSetOutputs, &offsetWeights[0], &offsetDeltas[0], &offsetOuts[0], &order[0], instance, params.errorFunc);

                //update the weights using the deltas
                //no momentum is used, it will be added after all the instances
                weightsUpdate(&values[0], weights, &tmpWeights[0], &deltas[0], numOfLayers, layersSize, &offsetIns[0], &offsetWeights[0], &offsetDeltas[0], 0, &oldWeights[0], params.learningRate);
            }



            //add temporary weights changes to real weights (the total is divided among the total number of instances (to use the same learning rate of the standard BP)
            //it also uses momentum
            for (int w = 0; w<numOfWeights; w++) {
                float auxWeight = weights[w];
                weights[w] += (tmpWeights[w] / numOfInstances) + params.momentum*(auxWeight - oldWeights[w]);
                tmpWeights[w] = 0;
                oldWeights[w] = auxWeight;
            }

            if (params.epochs_between_reports > 0 && epoch % params.epochs_between_reports == 0) {

                mseTrain = net->computeMSE(*trainingSet);
                if (printtype == PRINT_ALL)
                    printf("Epoch\t%d\tError on train set %.10f", epoch, mseTrain);

                if (testSet != NULL) {

                    mseTest = net->computeMSE(*testSet);
                    if (mseTest<bestMSETest) {
                        bestMSETest = mseTest;
                        if (bestMSETestNet) {
                            *bestMSETestNet = *net;
                        }
                    }
                    if ((mseTrain + mseTest)<bestMSETrainTest&&bestMSETrainTestNet) {
                        *bestMSETrainTestNet = *net;
                        bestMSETrainTest = mseTrain + mseTest;
                    }
                    if (printtype == PRINT_ALL)
                        printf("\t\tError on test set %.10f", mseTest);

                    if (bestClassTestNet) {
                        float per = net->classificatePerc(*testSet);
                        if (printtype == PRINT_ALL)
                            printf("\t\tClassification percentage on test set: %.1f%%", per * 100);
                        if (per>bestClassTest) {
                            *bestClassTestNet = *net;
                            bestClassTest = per;
                            if (printtype == PRINT_ALL)
                                printf(" ***");
                        }
                    }

                    if (mseTest <= params.desired_error) {
                        if (printtype == PRINT_ALL)
                            printf("\nDesired error reached on test set.\n");
                        break;
                    }

                }

                if (printtype == PRINT_ALL)
                    printf("\n");

                if (mseTrain <= params.desired_error && !testSet) {
                    if (printtype == PRINT_ALL)
                        printf("Desired error reached on training set.\n");
                    break;
                }
            }
        }


        if (printtype == PRINT_ALL)
            printf("Training complete.\n");
        if (testSet)
            return bestMSETest;
        else
            return mseTrain;

    }

#ifndef DISABLE_CUDA_NN
    ///batch training on device
    ///n is the number of parameters. parameters are (float array):
    ///desired error, max_epochs, epochs_between_reports, learning_rate, momentum (using momentum is 20% slower), shuffle (SHUFFLE_ON or SHUFFLE_OFF), error function (ERROR_TANH or ERROR_LINEAR)void FeedForwardNNTrainer::trainCpuBatch(const int n, const float * params){
    float trainGPUBatch(const TrainingParameters & params, const int printtype) {

        if (printtype != PRINT_OFF)
            print_parameters(params);

        float mseTrain = FLT_MAX, mseTest = FLT_MAX;
        //declare some error values for evaluating the trained network and storing best results
        //best net MSE on test, best net MSE on train+test, best net as classifier on test
        float bestMSETest = FLT_MAX, bestMSETrainTest = FLT_MAX, bestClassTest = 0;


        //declare some network values
        int numOfLayers = net->getNumOfLayers();
        int numOfWeights = net->getNumOfWeights();
        int numOfNeurons = 0;
        for (int i = 0; i < net->getNumOfLayers(); i++) {
            numOfNeurons += net->getLayersSize()[i] + 1;
        }
        //declare some training set values
        int numOfInstances = trainingSet->getNumOfInstances();
        int numOfInputsPerInstance = trainingSet->getNumOfInputsPerInstance();
        int numOfOutputsPerInstance = trainingSet->getNumOfOutputsPerInstance();

        int numOfTestInstances = 0;

        if (testSet)
            numOfTestInstances = testSet->getNumOfInstances();

        //declare an array of neurons to represent the neuron values
        std::vector<float> values(numOfNeurons*numOfInstances);
        //declare an array of neurons to represent the neuron values of the test set
        std::vector<float> testValues(numOfNeurons*numOfTestInstances);

        //declare an array of deltas to represent the gradients for the weight updates
        std::vector<float> deltas(numOfNeurons*numOfInstances);
        //training and test set to be uploaded in device memory
        std::vector<float> columnTrainingSetInputs(numOfInstances*numOfInputsPerInstance);
        std::vector<float> columnTrainingSetOutputs(numOfInstances*numOfOutputsPerInstance);
        std::vector<float> columnTestSetInputs(numOfTestInstances*numOfInputsPerInstance);
        std::vector<float> columnTestSetOutputs(numOfTestInstances*numOfOutputsPerInstance);
        //declare an array of weights to use for momentum
        std::vector<float> oldWeights(numOfWeights);
        //declare a pointer to the net weights
        float * weights = net->getWeights();
        //declare a pointer to the net activation functions
        const int * actFuncts = net->getActFuncts();
        //declare a pointer to the net layers size
        const int * layersSize = net->getLayersSize();

        //declare a pointer to the training set inputs
        float * trainingSetInputs = trainingSet->getInputs();
        //declare a pointer to the training set outputs
        float * trainingSetOutputs = trainingSet->getOutputs();

        //declare a pointer to the test set inputs
        float * testSetInputs = nullptr;
        //declare a pointer to the test set outputs
        float * testSetOutputs = nullptr;
        if (testSet) {
            testSetInputs = testSet->getInputs();
            testSetOutputs = testSet->getOutputs();
        }

        //retrieve some offsets to manage array indexes of each layer 'i'
        auto offsetWeights = net->get_weight_offsets(1);
        auto offsetIns = net->get_input_offsets(numOfInstances);
        auto offsetOuts = net->get_output_offsets(numOfInstances);

        auto offsetDeltas = net->get_output_offsets(numOfInstances);
        auto offsetTestIns = net->get_input_offsets(numOfTestInstances);
        auto offsetTestOuts = net->get_output_offsets(numOfTestInstances);

        
        //row-major->column major indexing
        for (int i = 0; i < numOfInstances; i++) {
            for (int j = 0; j < numOfInputsPerInstance; j++)
                columnTrainingSetInputs[j * numOfInstances + i] = trainingSetInputs[i * numOfInputsPerInstance + j];
            for (int j = 0; j < numOfOutputsPerInstance; j++)
                columnTrainingSetOutputs[j * numOfInstances + i] = trainingSetOutputs[i * numOfOutputsPerInstance + j];
        }

        for (int i = 0; i < numOfTestInstances; i++) {
            for (int j = 0; j < numOfInputsPerInstance; j++)
                columnTestSetInputs[j * numOfTestInstances + i] = testSetInputs[i * numOfInputsPerInstance + j];
            for (int j = 0; j < numOfOutputsPerInstance; j++)
                columnTestSetOutputs[j * numOfTestInstances + i] = testSetOutputs[i * numOfOutputsPerInstance + j];
        }

        //copy the training set into the input neurons values
        for (int i = 0; i < numOfInstances * numOfInputsPerInstance; i++)
            values[i] = columnTrainingSetInputs[i];

        //copy the test set into the input neurons values
        for (int i = 0; i < numOfTestInstances * numOfInputsPerInstance; i++)
            testValues[i] = columnTestSetInputs[i];

        //BIAS initializations
        for (int i = 0; i < numOfLayers; i++) {
            for (int j = offsetIns[i] + (layersSize[i]) * numOfInstances; j < offsetOuts[i]; j++)
                values[j] = 1.0f;
        }
        if (testSet)
            for (int i = 0; i < numOfLayers; i++) {
                for (int j = offsetTestIns[i] + (layersSize[i]) * numOfTestInstances; j < offsetTestOuts[i]; j++)
                    testValues[j] = 1.0f;
            }


        //vector to shuffle training set
        std::vector<int> order(numOfInstances);
        for (int i = 0; i < numOfInstances; i++)
            order[i] = i;


        //cublas initializations
        cublasStatus stat;

        cublasInit();

        float * devValues = nullptr;
        float * devTestValues = nullptr;
        float * devDeltas = nullptr;
        float * devWeights = nullptr;
        float * devOldWeights = nullptr;

        float * devTrainingSetInputs = nullptr;
        float * devTrainingSetOutputs = nullptr;
        float * devTestSetInputs = nullptr;
        float * devTestSetOutputs = nullptr;

        auto testAllocSuccess = [&]() {
            if (stat != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("device memory allocation failed");
        };
        //allocates the vectors on the device
        stat = cublasAlloc(numOfNeurons*numOfInstances, sizeof(values[0]), (void**)&devValues);
        testAllocSuccess();
        if (testSet) {
            stat = cublasAlloc(numOfNeurons*numOfTestInstances, sizeof(testValues[0]), (void**)&devTestValues);
            testAllocSuccess();
        }
        stat = cublasAlloc(numOfNeurons*numOfInstances, sizeof(deltas[0]), (void**)&devDeltas);
        testAllocSuccess();
        stat = cublasAlloc(numOfWeights, sizeof(*weights), (void**)&devWeights);
        testAllocSuccess();
        stat = cublasAlloc(numOfWeights, sizeof(oldWeights[0]), (void**)&devOldWeights);
        testAllocSuccess();

        stat = cublasAlloc(numOfInstances*numOfInputsPerInstance, sizeof(*devTrainingSetInputs), (void**)&devTrainingSetInputs);
        testAllocSuccess();
        stat = cublasAlloc(numOfInstances*numOfOutputsPerInstance, sizeof(*devTrainingSetOutputs), (void**)&devTrainingSetOutputs);
        testAllocSuccess();
        if (testSet) {
            stat = cublasAlloc(numOfTestInstances*numOfInputsPerInstance, sizeof(*devTestSetInputs), (void**)&devTestSetInputs);
            testAllocSuccess();
            stat = cublasAlloc(numOfTestInstances*numOfOutputsPerInstance, sizeof(*devTestSetOutputs), (void**)&devTestSetOutputs);
            testAllocSuccess();
        }

        //copies the training set inputs and outputs on the device
        cudaMemcpy(devTrainingSetInputs, &columnTrainingSetInputs[0], numOfInstances*numOfInputsPerInstance * sizeof(columnTrainingSetInputs[0]), cudaMemcpyHostToDevice);
        cudaMemcpy(devTrainingSetOutputs, &columnTrainingSetOutputs[0], numOfInstances*numOfOutputsPerInstance * sizeof(columnTrainingSetOutputs[0]), cudaMemcpyHostToDevice);

        if (testSet) {
            //copies the test set inputs and outputs on the device
            cudaMemcpy(devTestSetInputs, &columnTestSetInputs[0], numOfTestInstances*numOfInputsPerInstance * sizeof(columnTestSetInputs[0]), cudaMemcpyHostToDevice);
            cudaMemcpy(devTestSetOutputs, &columnTestSetOutputs[0], numOfTestInstances*numOfOutputsPerInstance * sizeof(columnTestSetOutputs[0]), cudaMemcpyHostToDevice);
        }

        //copies the training set inputs with the biases and the weights to the device
        cudaMemcpy(devValues, &values[0], numOfNeurons*numOfInstances * sizeof(values[0]), cudaMemcpyHostToDevice);

        if (testSet) {
            //copies the test set inputs with the biases and the weights to the device
            cudaMemcpy(devTestValues, &testValues[0], numOfNeurons*numOfTestInstances * sizeof(testValues[0]), cudaMemcpyHostToDevice);
        }

        cudaMemcpy(devDeltas, &deltas[0], numOfNeurons*numOfInstances * sizeof(deltas[0]), cudaMemcpyHostToDevice);
        //weights are allocated row-major
        cudaMemcpy(devWeights, weights, numOfWeights * sizeof(*weights), cudaMemcpyHostToDevice);
        cudaMemcpy(devOldWeights, weights, numOfWeights * sizeof(*weights), cudaMemcpyHostToDevice);

        if (printtype == PRINT_ALL) {
            //compute starting error rates (GPU)
            printf("Starting:\tError on train set %.10f", GPUComputeMSE(devValues, devWeights, actFuncts, numOfLayers, layersSize, numOfInstances, numOfOutputsPerInstance, devTrainingSetOutputs, &offsetIns[0], &offsetWeights[0], &offsetOuts[0]));
            if (testSet) {
                printf("\t\tError on test set %.10f", GPUComputeMSE(devTestValues, devWeights, actFuncts, numOfLayers, layersSize, numOfTestInstances, numOfOutputsPerInstance, devTestSetOutputs, &offsetTestIns[0], &offsetWeights[0], &offsetTestOuts[0]));
            }
            printf("\n");
        }

        //epochs training
        for (int epoch = 1; epoch <= params.max_epochs && quit == false; epoch++) {

            //shuffle instances
            int ind = 0, aux = 0;
            if (params.shuff == SHUFFLE_ON)
                for (int i = 0; i<numOfInstances; i++) {
                    ind = gen_random_int(i, numOfInstances - 1);
                    aux = order[ind];
                    order[ind] = order[i];
                    order[i] = aux;
                }

            GPUForward(devValues, devWeights,
                actFuncts, numOfLayers, layersSize, numOfInstances, numOfInstances,
                &offsetIns[0], &offsetWeights[0], &offsetOuts[0]);

            //computes all the instances backward of the backpropagation training
            GPUBack(devValues, devWeights, devDeltas,
                actFuncts, numOfLayers, layersSize, numOfInstances, numOfInstances, numOfOutputsPerInstance,
                devTrainingSetOutputs, &offsetWeights[0], &offsetDeltas[0], &offsetOuts[0],
                params.errorFunc);

            //update the weights using the deltas
            GPUUpdate(devValues, devWeights, devDeltas,
                numOfLayers, layersSize, numOfInstances, numOfInstances,
                &offsetIns[0], &offsetWeights[0], &offsetDeltas[0],
                params.momentum, devOldWeights, params.learningRate);

            if (params.epochs_between_reports > 0 && epoch % params.epochs_between_reports == 0) {

                cudaMemcpy(weights, devWeights, numOfWeights * sizeof(float), cudaMemcpyDeviceToHost);

                //float mseTrain=net->computeMSE(*trainingSet);
                mseTrain = GPUComputeMSE(devValues, devWeights, actFuncts, numOfLayers, layersSize, numOfInstances, numOfOutputsPerInstance, devTrainingSetOutputs, &offsetIns[0], &offsetWeights[0], &offsetOuts[0]);
                if (printtype == PRINT_ALL)
                    printf("Epoch    %d    Error on train set %.10f", epoch, mseTrain);

                if (testSet) {

                    //float mseTest=net->computeMSE(*testSet);
                    mseTest = GPUComputeMSE(devTestValues, devWeights, actFuncts, numOfLayers, layersSize, numOfTestInstances, numOfOutputsPerInstance, devTestSetOutputs, &offsetTestIns[0], &offsetWeights[0], &offsetTestOuts[0]);
                    if (mseTest<bestMSETest) {
                        bestMSETest = mseTest;
                        if (bestMSETestNet) {
                            *bestMSETestNet = *net;
                        }
                    }
                    if ((mseTrain + mseTest)<bestMSETrainTest&&bestMSETrainTestNet) {
                        *bestMSETrainTestNet = *net;
                        bestMSETrainTest = mseTrain + mseTest;
                    }
                    if (printtype == PRINT_ALL)
                        printf("        Error on test set %.10f", mseTest);

                    if (bestClassTestNet) {
                        //float per=net->classificatePerc(*testSet);
                        float per = GPUclassificatePerc(devTestValues, devWeights, actFuncts, numOfLayers, layersSize, numOfTestInstances, numOfOutputsPerInstance, devTestSetOutputs, &offsetTestIns[0], &offsetWeights[0], &offsetTestOuts[0]);
                        if (printtype == PRINT_ALL)
                            printf("        Classification percentage on test set: %.1f%%", per * 100);
                        if (per>bestClassTest) {
                            *bestClassTestNet = *net;
                            bestClassTest = per;
                            if (printtype == PRINT_ALL)
                                printf(" ***");
                        }
                    }

                    if (mseTest <= params.desired_error) {
                        if (printtype == PRINT_ALL)
                            printf("\nDesired error reached on test set.\n");
                        break;
                    }

                }

                if (printtype == PRINT_ALL)
                    printf("\n");

                if (mseTrain <= params.desired_error && !testSet) {
                    if (printtype == PRINT_ALL)
                        printf("Desired error reached on training set.\n");
                    break;
                }
            }
        }

        cudaMemcpy(weights, devWeights, numOfWeights * sizeof(float), cudaMemcpyDeviceToHost);

        //cublas deallocations
        cublasFree(devValues);
        cublasFree(devTestValues);
        cublasFree(devDeltas);
        cublasFree(devWeights);
        cublasFree(devOldWeights);

        cublasFree(devTrainingSetInputs);
        cublasFree(devTrainingSetOutputs);
        cublasFree(devTestSetInputs);
        cublasFree(devTestSetOutputs);

        cublasShutdown();

        if (printtype == PRINT_ALL)
            printf("Training complete.\n");
        if (testSet)
            return bestMSETest;
        else
            return mseTrain;
    }
#endif

    ///computes a single instance forward of the backpropagation training
    void stepForward(float * values, const  float * weights, const  int * actFuncts, const  int numOfLayers, const  int * layersSize, const  int numOfInputsPerInstance, const float * trainingSetInputs, const int * offsetIns, const int * offsetWeights, const int * offsetOuts, const int * order, const int instance) {
        //load an array of inputs
        for (int i = 0; i<numOfInputsPerInstance; i++)
            values[i] = trainingSetInputs[order[instance] * numOfInputsPerInstance + i];

        //loops the layers
        for (int i = 0; i<numOfLayers - 1; i++) {

            //bias neuron
            values[offsetIns[i] + layersSize[i]] = 1.0;

            float tot = 0;
            //loops the outputs
            for (int j = 0; j<layersSize[i + 1]; j++) {
                //unrolled sum of all to avoid some floating points precision problems
                tot = 0;
                int k = (layersSize[i] + 1) % 4;
                switch (k) {
                    case 3:tot += weights[2 + j*(layersSize[i] + 1) + offsetWeights[i]] * values[2 + offsetIns[i]];
                    case 2:tot += weights[1 + j*(layersSize[i] + 1) + offsetWeights[i]] * values[1 + offsetIns[i]];
                    case 1:tot += weights[j*(layersSize[i] + 1) + offsetWeights[i]] * values[offsetIns[i]];
                    case 0:break;
                }
                for (; k<layersSize[i] + 1; k += 4) {
                    tot += weights[k + j*(layersSize[i] + 1) + offsetWeights[i]] * values[k + offsetIns[i]] +
                        weights[k + 1 + j*(layersSize[i] + 1) + offsetWeights[i]] * values[k + 1 + offsetIns[i]] +
                        weights[k + 2 + j*(layersSize[i] + 1) + offsetWeights[i]] * values[k + 2 + offsetIns[i]] +
                        weights[k + 3 + j*(layersSize[i] + 1) + offsetWeights[i]] * values[k + 3 + offsetIns[i]];
                }
                //write the ouputs of the layer
                values[j + offsetOuts[i]] = actFunction(actFuncts[i + 1], tot);
            }
        }
    }

    ///computes a single instance backward of the backpropagation training
    void stepBack(const float * values, const  float * weights, float * deltas, const  int * actFuncts, const  int numOfLayers, const  int * layersSize, const  int numOfOutputsPerInstance, const float * trainingSetOutputs, const int * offsetWeights, const int * offsetDeltas, const int * offsetOuts, const int * order, const int instance, const int errorFunc) {
        //loop layers backwards (from last hidden to inputs)
        for (int i = numOfLayers - 2; i >= 0; i--) {
            //output layer (different rule) and no bias (for nextLayerSize)
            if (i == numOfLayers - 2) {
                for (int j = 0; j<layersSize[i + 1]; j++) {
                    float error = (trainingSetOutputs[j + order[instance] * numOfOutputsPerInstance] - values[j + offsetOuts[i]]) / spanSize(actFuncts[i + 1]);
                    deltas[j + offsetDeltas[i]] = actDerivation(actFuncts[i + 1], values[j + offsetOuts[i]])*errorFunction(error, errorFunc);
                }
            }
            //normal hidden layer
            else {
                //new efficent code
                std::vector<float> tmpErrors(layersSize[i + 1] + 1);
                for (int j = 0; j<layersSize[i + 1] + 1; j++)
                    tmpErrors[j] = 0;

                for (int k = 0; k<layersSize[i + 2]; k++) {
                    float precalc = deltas[k + offsetDeltas[i + 1]];
                    for (int j = 0; j<layersSize[i + 1] + 1; j++) {
                        //next layer's delta and weights are used
                        tmpErrors[j] += precalc*weights[j + k*(layersSize[i + 1] + 1) + offsetWeights[i + 1]];
                    }
                }

                for (int j = 0; j<layersSize[i + 1] + 1; j++) {
                    deltas[j + offsetDeltas[i]] = actDerivation(actFuncts[i + 1], values[j + offsetOuts[i]])*tmpErrors[j];
                }
            }
        }
    }

    ///update the weights using the deltas
    void weightsUpdate(const float * values, const float * weights, float * weightsToUpdate, const float * deltas, const  int numOfLayers, const  int * layersSize, const int * offsetIns, const int * offsetWeights, const int * offsetDeltas, const float momentum, float * oldWeights, float learningRate) {
        //loops the layers
        if (momentum>0)
            for (int i = 0; i<numOfLayers - 1; i++) {
                for (int k = 0; k<layersSize[i + 1]; k++) {
                    //efficient code to speed up the backpropagation
                    float tempLxD = learningRate*deltas[k + offsetDeltas[i]];
                    int wOffset = k*(layersSize[i] + 1) + offsetWeights[i];
                    int vOffset = offsetIns[i];
                    for (int j = 0; j<layersSize[i] + 1; j++) {
                        float auxWeight = weights[j + wOffset];
                        weightsToUpdate[j + wOffset] += tempLxD*values[j + vOffset] + momentum*(auxWeight - oldWeights[j + wOffset]);
                        oldWeights[j + wOffset] = auxWeight;
                    }
                }
            }
        else
            for (int i = 0; i<numOfLayers - 1; i++) {
                for (int k = 0; k<layersSize[i + 1]; k++) {
                    //efficient code to speed up the backpropagation
                    float tempLxD = learningRate*deltas[k + offsetDeltas[i]];
                    int wOffset = k*(layersSize[i] + 1) + offsetWeights[i];
                    int vOffset = offsetIns[i];
                    for (int j = 0; j<layersSize[i] + 1; j++) {
                        weightsToUpdate[j + wOffset] += tempLxD*values[j + vOffset];
                    }
                }
            }



    }

#ifndef DISABLE_CUDA_NN
    ///GPU computes all the instances forward of the backpropagation training
    void GPUForward(
        float * devValues, const  float * devWeights,
        const  int * actFuncts, const  int numOfLayers, const  int * layersSize,
        const int totNumOfInstances, const int numOfInstancesToUse,
        const int * offsetIns, const int * offsetWeights, const int * offsetOuts) {
        //loops the layers
        for (int i = 0; i<numOfLayers - 1; i++) {

            int ninput = totNumOfInstances;
            int naux = layersSize[i] + 1;
            int noutput = layersSize[i + 1];

            const float * devPtrA;
            const float * devPtrB;
            float * devPtrC;
            devPtrA = devValues + offsetIns[i];
            devPtrB = devWeights + offsetWeights[i];
            devPtrC = devValues + offsetOuts[i];

            //does the product of the neurons matrix and the weights matrix
            //the weights matrix is row-major so no translation is necessary
            cublasSgemm('n', 'n',
                numOfInstancesToUse, noutput, naux,
                1, devPtrA, ninput,
                devPtrB, naux,
                0, devPtrC, ninput
            );

            computeActFunct(devPtrC, numOfInstancesToUse*noutput, actFuncts[i + 1]);

        }

    }

    ///GPU computes all the instances backward of the backpropagation training
    void GPUBack(
        const float * devValues, const float * devWeights, float * devDeltas,
        const int * actFuncts, const int numOfLayers, const int *layersSize,
        const int totNumOfInstances, const int numOfInstancesToUse, const int numOfOutputsPerInstance,
        const float * devTrainingSetOutputs, const int *offsetWeights, const int *offsetDeltas, const int * offsetOuts, const int errorFunc) {
        //loop layers backwards (from last hidden to inputs)
        for (int i = numOfLayers - 2; i >= 0; i--) {
            //output layer (different rule) and no bias (for nextLayerSize)
            if (i == numOfLayers - 2) {
                computeError(devDeltas + offsetDeltas[i], devTrainingSetOutputs, devValues + offsetOuts[i], numOfInstancesToUse*numOfOutputsPerInstance, actFuncts[i + 1], errorFunc);
            }
            //normal hidden layer
            else {
                int ninput = totNumOfInstances;
                int naux = layersSize[i + 2];
                int noutput = layersSize[i + 1] + 1;

                const float * devPtrA;
                const float * devPtrB;
                float * devPtrC;
                devPtrA = devDeltas + offsetDeltas[i + 1];
                devPtrB = devWeights + offsetWeights[i + 1];
                devPtrC = devDeltas + offsetDeltas[i];

                //does the product of the deltas matrix and the weights matrix
                //the weights matrix is row-major so must be translated to multiply. also the index is noutput
                cublasSgemm('n', 't',
                    numOfInstancesToUse, noutput, naux,
                    1, devPtrA, ninput,
                    devPtrB, noutput,
                    0, devPtrC, ninput
                );
            }
            computeDerivFunct(devDeltas + offsetDeltas[i], devValues + offsetOuts[i], numOfInstancesToUse*layersSize[i + 1], actFuncts[i + 1]);
        }
    }

    ///GPU updates the weights for all the instances
    void GPUUpdate(
        const float * devValues, float * devWeights, const float *devDeltas,
        const int numOfLayers, const int * layersSize,
        const int totNumOfInstances, const int numOfInstancesToUse,
        const int * offsetIns, const int * offsetWeights, const int * offsetDeltas,
        const float momentum, float * devOldWeights, const float learningRate) {

        //loops the layers
        for (int i = 0; i<numOfLayers - 1; i++) {

            int ninput = layersSize[i] + 1;
            int naux = totNumOfInstances;
            int noutput = layersSize[i + 1];

            const float * devPtrA;
            const float * devPtrB;
            float * devPtrC;
            devPtrA = devValues + offsetIns[i];
            devPtrB = devDeltas + offsetDeltas[i];
            devPtrC = devWeights + offsetWeights[i];

            if (momentum>0) {
                //if there's a momentum it updates the weights with a portion of the difference with the old weights
                addMomentum(devWeights + offsetWeights[i], devOldWeights + offsetWeights[i], layersSize[i + 1] * (layersSize[i] + 1), momentum);
            }

            //does the product of neurons matrix and the deltas matrix and add them to weights matrix (after multiplying with learning rate and dividing by nOfIstances)
            //the neurons matrix is translated to multiply
            cublasSgemm('t', 'n',
                ninput, noutput, numOfInstancesToUse,
                learningRate / (float)totNumOfInstances, devPtrA, naux,
                devPtrB, naux,
                1, devPtrC, ninput
            );

        }

    }

    ///GPU computes the MSE on a set
    float GPUComputeMSE(float * devValues, const  float * devWeights, const  int * actFuncts, const  int numOfLayers, const  int * layersSize, const int numOfInstances, const int numOfOutputsPerInstance, const float * devSetOutputs, const int * offsetIns, const int * offsetWeights, const int * offsetOuts) {

        //loops the layers
        for (int i = 0; i<numOfLayers - 1; i++) {

            int ninput = numOfInstances;
            int naux = layersSize[i] + 1;
            int noutput = layersSize[i + 1];

            const float * devPtrA;
            const float * devPtrB;
            float * devPtrC;
            devPtrA = devValues + offsetIns[i];
            devPtrB = devWeights + offsetWeights[i];
            devPtrC = devValues + offsetOuts[i];

            //does the product of the neurons matrix and the weights matrix
            //the weights matrix is row-major so no translation is necessary
            cublasSgemm('n', 'n',
                ninput, noutput, naux,
                1, devPtrA, ninput,
                devPtrB, naux,
                0, devPtrC, ninput
            );

            computeActFunct(devPtrC, ninput*noutput, actFuncts[i + 1]);

        }

        return mseError(devSetOutputs, devValues + offsetOuts[numOfLayers - 2], numOfInstances*numOfOutputsPerInstance, actFuncts[numOfLayers - 1]);

    }

    ///GPU computes the classification percentage on a set
    float GPUclassificatePerc(float * devValues, const  float * devWeights, const  int * actFuncts, const  int numOfLayers, const  int * layersSize, const int numOfInstances, const int numOfOutputsPerInstance, float * devSetOutputs, const int * offsetIns, const int * offsetWeights, const int * offsetOuts) {

        //loops the layers
        for (int i = 0; i<numOfLayers - 1; i++) {

            int ninput = numOfInstances;
            int naux = layersSize[i] + 1;
            int noutput = layersSize[i + 1];

            const float * devPtrA;
            const float * devPtrB;
            float * devPtrC;
            devPtrA = devValues + offsetIns[i];
            devPtrB = devWeights + offsetWeights[i];
            devPtrC = devValues + offsetOuts[i];

            //does the product of the neurons matrix and the weights matrix
            //the weights matrix is row-major so no translation is necessary
            cublasSgemm('n', 'n',
                ninput, noutput, naux,
                1, devPtrA, ninput,
                devPtrB, naux,
                0, devPtrC, ninput
            );

            computeActFunct(devPtrC, ninput*noutput, actFuncts[i + 1]);

        }


        std::vector<int> valuesIndexes(numOfInstances);
        std::vector<int> outputIndexes(numOfInstances);
        int * devValuesIndexes;
        int * devOutputIndexes;
        cudaMalloc((void **)&devValuesIndexes, numOfInstances * sizeof(int));
        cudaMalloc((void **)&devOutputIndexes, numOfInstances * sizeof(int));

        float * tmpTranslate;
        cudaMalloc((void **)&tmpTranslate, numOfInstances*numOfOutputsPerInstance * sizeof(int));

        //translate the output neurons matrix from column major to row major
        translateMatrix(numOfInstances, numOfOutputsPerInstance, devValues + offsetOuts[numOfLayers - 2], tmpTranslate);
        //and evaluates the max of each row for classification
        computeMaxes(numOfInstances, numOfOutputsPerInstance, tmpTranslate, devValuesIndexes);

        //translate the desired outputs matrix from column major to row major
        translateMatrix(numOfInstances, numOfOutputsPerInstance, devSetOutputs, tmpTranslate);
        //and evaluates the max of each row for classification
        computeMaxes(numOfInstances, numOfOutputsPerInstance, tmpTranslate, devOutputIndexes);

        cudaFree(tmpTranslate);


        cudaMemcpy(&valuesIndexes[0], devValuesIndexes, numOfInstances * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&outputIndexes[0], devOutputIndexes, numOfInstances * sizeof(int), cudaMemcpyDeviceToHost);

        //compute the actual rate comparing the correct classification and the one of the net
        int cont = 0;
        for (int i = 0; i<numOfInstances; i++) {
            if (valuesIndexes[i] == outputIndexes[i])cont++;
        }

        cudaFree(devValuesIndexes);
        cudaFree(devOutputIndexes);
        return (float)cont / (float)numOfInstances;
    }
#endif

    FeedForwardNN * net = nullptr;
    LearningSet * trainingSet = nullptr;
    LearningSet * testSet = nullptr;
    FeedForwardNN * bestMSETestNet = nullptr;
    FeedForwardNN * bestMSETrainTestNet = nullptr;
    FeedForwardNN * bestClassTestNet = nullptr;
};
