/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

#pragma once


#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <vector>

#include "FeedForwardNN.h"
#include "LearningSet.h"
#include "ErrorFunctions.h"
#ifdef USE_CUDA
#include "CudaActivationFunctions.cuh"
#include "CudaErrorFunctions.cuh"
#endif

#include "RandomGenerator.h"

#ifdef USE_CUDA
#include <cublas.h>
//#include <cutil_inline.h>
#endif


#define TRAIN_CPU 0
#ifdef USE_CUDA
#define TRAIN_GPU 1
#endif
#define ALG_BP 0
#define ALG_BATCH 1
#define SHUFFLE_OFF 0
#define SHUFFLE_ON 1

#define PRINT_ALL 0
#define PRINT_MIN 1
#define PRINT_OFF 2

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
    float train(const int n, const float * params, const int printtype = PRINT_ALL) {
        //checks CTRL-C to interrupt training manually
        quit = false;
       // signal(SIGINT, terminate);

        setvbuf(stdout, (char*)NULL, _IONBF, 0);

        // checks for network and training set correct initialization
        if (net == NULL) { printf("NEURAL NETWORK NOT SELECTED\n"); exit(1); }
        if (trainingSet == NULL) { printf("TRAINING SET NOT SELECTED\n"); exit(1); }
        if ((trainingSet->getNumOfInputsPerInstance() != net->getLayersSize()[0])
            || (trainingSet->getNumOfOutputsPerInstance() != net->getLayersSize()[net->getNumOfLayers() - 1])) {
            printf("NETWORK AND TRAINING SET OF DIFFERENT SIZE\n"); exit(1);
        }
        if (testSet != NULL &&
            (trainingSet->getNumOfInputsPerInstance() != testSet->getNumOfInputsPerInstance()
                || trainingSet->getNumOfOutputsPerInstance() != testSet->getNumOfOutputsPerInstance())) {
            printf("TEST SET OF DIFFERENT SIZE\n"); exit(1);
        }
        if (n<1) { printf("TOO FEW PARAMETERS SELECTED FOR TRAINING\n"); exit(1); }

        if (printtype != PRINT_OFF) {
            printf("Network:\t\t");
            printf("%d", net->getLayersSize()[0]);
            for (int i = 1; i<net->getNumOfLayers(); i++)
                printf("x%d", net->getLayersSize()[i]);
            printf("\n");
            printf("Activation functions:\t");
            for (int i = 0; i<net->getNumOfLayers(); i++)
                printf("%d ", net->getActFuncts()[i]);
            printf("\n");
        }

        //select the right algorithm to execute training
        switch ((int)params[0]) {
            case TRAIN_CPU:
                switch ((int)params[1]) {
                    case ALG_BP:        return trainCpuBp(n - 2, params + 2, printtype);                    break;
                    case ALG_BATCH:     return trainCpuBatch(n - 2, params + 2, printtype);                break;
                    default:            printf("TRAINING NOT IMPLEMENTED YET\n"); exit(1);    break;
                }
                break;
#ifdef USE_CUDA
            case TRAIN_GPU:
                switch ((int)params[1]) {
                    case ALG_BP:        printf("TRAINING NOT IMPLEMENTED YET\n"); exit(1);    break;
                    case ALG_BATCH:     return trainGPUBatch(n - 2, params + 2, printtype);        break;
                    default:            printf("TRAINING NOT IMPLEMENTED YET\n"); exit(1);    break;
                }
                break;
#endif
            default:printf("TRAINING NOT IMPLEMENTED YET\n"); exit(1); break;
        }

        //stops checking CTRL-C
        signal(SIGINT, SIG_DFL);
}

private:
    ///backpropagation training on host
    ///n is the number of parameters. parameters are (float array):
    ///desired error, max_epochs, epochs_between_reports, learning_rate, momentum (using momentum is 20% slower), shuffle (SHUFFLE_ON or SHUFFLE_OFF), error function (ERROR_TANH or ERROR_LINEAR)
    float trainCpuBp(const int n, const float * params, const int printtype) {
        //parameters parsing
        float desired_error;
        int max_epochs;
        int epochs_between_reports;
        float learningRate;
        float momentum;
        int shuff;
        int errorFunc;

        if (n<2) { printf("TOO FEW PARAMETERS SELECTED FOR TRAINING\n"); exit(1); }
        desired_error = params[0];
        max_epochs = params[1];
        if (n >= 3)
            epochs_between_reports = params[2];
        else
            epochs_between_reports = max_epochs / 10;
        if (n >= 4)
            learningRate = params[3];
        else
            learningRate = 0.7;
        if (n >= 5)
            momentum = params[4];
        else
            momentum = 0;
        if (n >= 6)
            shuff = params[5];
        else
            shuff = SHUFFLE_ON;
        if (n >= 7)
            errorFunc = params[6];
        else
            errorFunc = ERROR_TANH;

        if (printtype != PRINT_OFF) {
            printf("Training on:\t\tCPU\n");
            printf("Algorithm:\t\tBackpropagation\n");
            printf("Desired Error:\t\t%f\n", desired_error);
            printf("Max epochs:\t\t%d\n", max_epochs);
            printf("Epochs between reports:\t%d\n", epochs_between_reports);
            printf("Learning rate:\t\t%f\n", learningRate);
            printf("Momentum:\t\t%f\n", momentum);
            if (shuff == SHUFFLE_ON)
                printf("Shuffle:\t\tON\n");
            else
                printf("Shuffle:\t\tOFF\n");
            if (errorFunc == ERROR_TANH)
                printf("Error function:\t\tTANH\n");
            else
                printf("Error function:\t\tLINEAR\n");
            printf("\n");
        }

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
        float * weights;
        weights = net->getWeights();
        //declare a pointer to the net activation functions
        const int * actFuncts;
        actFuncts = net->getActFuncts();
        //declare a pointer to the net layers size
        const int * layersSize;
        layersSize = net->getLayersSize();

        //declare some offsets to manage array indexes of each layer 'i'
        std::vector<int> offsetWeights(numOfLayers);
        std::vector<int> offsetIns(numOfLayers);
        std::vector<int> offsetOuts(numOfLayers);
        std::vector<int> offsetDeltas(numOfLayers);
        for (int i = 0; i<numOfLayers; i++) {
            //calculates the offsets of the arrays
            offsetWeights[i] = 0;
            offsetDeltas[i] = layersSize[0] + 1;
            offsetIns[i] = 0;
            offsetOuts[i] = layersSize[0] + 1;
            for (int j = 0; j<i; j++) {
                offsetWeights[i] += (layersSize[j] + 1)*layersSize[j + 1];
                offsetIns[i] += layersSize[j] + 1;
                offsetOuts[i] += layersSize[j + 1] + 1;
                offsetDeltas[i] += layersSize[j + 1] + 1;
            }
        }


        //save previous weights to use in momentum calculation
        for (int w = 0; w<numOfWeights; w++)
            oldWeights[w] = weights[w];


        //declare some training set values
        int numOfInstances = trainingSet->getNumOfInstances();
        int numOfInputsPerInstance = trainingSet->getNumOfInputsPerInstance();
        int numOfOutputsPerInstance = trainingSet->getNumOfOutputsPerInstance();

        //declare a pointer to the training set inputs
        float * trainingSetInputs;
        trainingSetInputs = trainingSet->getInputs();
        //declare a pointer to the training set outputs
        float * trainingSetOutputs;
        trainingSetOutputs = trainingSet->getOutputs();


        //vector to shuffle training set
        std::vector<int> order(numOfInstances);
        for (int i = 0; i<numOfInstances; i++)
            order[i] = i;

        if (printtype == PRINT_ALL) {
            //compute starting error rates
            printf("Starting:\tError on train set %.10f", net->computeMSE(*trainingSet));
            if (testSet != NULL) {
                printf("\t\tError on test set %.10f", net->computeMSE(*testSet));
            }
            printf("\n");
        }

        //epochs training
        for (int epoch = 1; epoch <= max_epochs&&quit == false; epoch++) {

            //shuffle instances
            int ind = 0, aux = 0;
            if (shuff == SHUFFLE_ON)
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
                stepBack(&values[0], weights, &deltas[0], actFuncts, numOfLayers, layersSize, numOfOutputsPerInstance, trainingSetOutputs, &offsetWeights[0], &offsetDeltas[0], &offsetOuts[0], &order[0], instance, errorFunc);

                //update the weights using the deltas
                weightsUpdate(&values[0], weights, weights, &deltas[0], numOfLayers, layersSize, &offsetIns[0], &offsetWeights[0], &offsetDeltas[0], momentum, &oldWeights[0], learningRate);

            }

            if (epochs_between_reports>0 && epoch%epochs_between_reports == 0) {

                mseTrain = net->computeMSE(*trainingSet);
                if (printtype == PRINT_ALL)
                    printf("Epoch\t%d\tError on train set %.10f", epoch, mseTrain);

                if (testSet != NULL) {

                    mseTest = net->computeMSE(*testSet);
                    if (mseTest<bestMSETest) {
                        bestMSETest = mseTest;
                        if (bestMSETestNet != NULL) {
                            *bestMSETestNet = *net;
                        }
                    }
                    if ((mseTrain + mseTest)<bestMSETrainTest&&bestMSETrainTestNet != NULL) {
                        *bestMSETrainTestNet = *net;
                        bestMSETrainTest = mseTrain + mseTest;
                    }
                    if (printtype == PRINT_ALL)
                        printf("\t\tError on test set %.10f", mseTest);

                    if (bestClassTestNet != NULL) {
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

                    if (mseTest <= desired_error) {
                        if (printtype == PRINT_ALL)
                            printf("\nDesired error reached on test set.\n");
                        break;
                    }

                }

                if (printtype == PRINT_ALL)
                    printf("\n");

                if (mseTrain <= desired_error&&testSet == NULL) {
                    if (printtype == PRINT_ALL)
                        printf("Desired error reached on training set.\n");
                    break;
                }
            }
        }

        if (printtype == PRINT_ALL)
            printf("Training complete.\n");
        if (testSet != NULL) {
            return bestMSETest;
        }
        else return mseTrain;

    }

    ///batch training on host
    ///n is the number of parameters. parameters are (float array):
    ///desired error, max_epochs, epochs_between_reports, learning_rate, momentum (using momentum is 20% slower), shuffle (SHUFFLE_ON or SHUFFLE_OFF), error function (ERROR_TANH or ERROR_LINEAR)void FeedForwardNNTrainer::trainCpuBatch(const int n, const float * params){
    float trainCpuBatch(const int n, const float * params, const int printtype) {
        //parameters parsing
        float desired_error;
        int max_epochs;
        int epochs_between_reports;
        float learningRate;
        float momentum;
        int shuff;
        int errorFunc;

        if (n<2) { printf("TOO FEW PARAMETERS SELECTED FOR TRAINING\n"); exit(1); }
        desired_error = params[0];
        max_epochs = params[1];
        if (n >= 3)
            epochs_between_reports = params[2];
        else
            epochs_between_reports = max_epochs / 10;
        if (n >= 4)
            learningRate = params[3];
        else
            learningRate = 0.7;
        if (n >= 5)
            momentum = params[4];
        else
            momentum = 0;
        if (n >= 6)
            shuff = params[5];
        else
            shuff = SHUFFLE_ON;
        if (n >= 7)
            errorFunc = params[6];
        else
            errorFunc = ERROR_TANH;

        if (printtype != PRINT_OFF) {
            printf("Training on:\t\tCPU\n");
            printf("Algorithm:\t\tBatch\n");
            printf("Desired Error:\t\t%f\n", desired_error);
            printf("Max epochs:\t\t%d\n", max_epochs);
            printf("Epochs between reports:\t%d\n", epochs_between_reports);
            printf("Learning rate:\t\t%f\n", learningRate);
            printf("Momentum:\t\t%f\n", momentum);
            if (shuff == SHUFFLE_ON)
                printf("Shuffle:\t\tON\n");
            else
                printf("Shuffle:\t\tOFF\n");
            if (errorFunc == ERROR_TANH)
                printf("Error function:\t\tTANH\n");
            else
                printf("Error function:\t\tLINEAR\n");
            printf("\n");
        }

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
        float * weights;
        weights = net->getWeights();
        //declare a pointer to the net activation functions
        const int * actFuncts;
        actFuncts = net->getActFuncts();
        //declare a pointer to the net layers size
        const int * layersSize;
        layersSize = net->getLayersSize();

        //declare some offsets to manage array indexes of each layer 'i'
        std::vector<int> offsetWeights(numOfLayers);
        std::vector<int> offsetIns(numOfLayers);
        std::vector<int> offsetOuts(numOfLayers);
        std::vector<int> offsetDeltas(numOfLayers);
        for (int i = 0; i<numOfLayers; i++) {
            //calculates the offsets of the arrays
            offsetWeights[i] = 0;
            offsetDeltas[i] = layersSize[0] + 1;
            offsetIns[i] = 0;
            offsetOuts[i] = layersSize[0] + 1;
            for (int j = 0; j<i; j++) {
                offsetWeights[i] += (layersSize[j] + 1)*layersSize[j + 1];
                offsetIns[i] += layersSize[j] + 1;
                offsetOuts[i] += layersSize[j + 1] + 1;
                offsetDeltas[i] += layersSize[j + 1] + 1;
            }
        }


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
        float * trainingSetInputs;
        trainingSetInputs = trainingSet->getInputs();
        //declare a pointer to the training set outputs
        float * trainingSetOutputs;
        trainingSetOutputs = trainingSet->getOutputs();


        //vector to shuffle training set
        std::vector<int> order(numOfInstances);
        for (int i = 0; i<numOfInstances; i++)
            order[i] = i;

        if (printtype == PRINT_ALL) {
            //compute starting error rates
            printf("Starting:\tError on train set %.10f", net->computeMSE(*trainingSet));
            if (testSet != NULL) {
                printf("\t\tError on test set %.10f", net->computeMSE(*testSet));
            }
            printf("\n");
        }

        //epochs training
        for (int epoch = 1; epoch <= max_epochs&&quit == false; epoch++) {

            //shuffle instances
            int ind = 0, aux = 0;
            if (shuff == SHUFFLE_ON)
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
                stepBack(&values[0], weights, &deltas[0], actFuncts, numOfLayers, layersSize, numOfOutputsPerInstance, trainingSetOutputs, &offsetWeights[0], &offsetDeltas[0], &offsetOuts[0], &order[0], instance, errorFunc);

                //update the weights using the deltas
                //no momentum is used, it will be added after all the instances
                weightsUpdate(&values[0], weights, &tmpWeights[0], &deltas[0], numOfLayers, layersSize, &offsetIns[0], &offsetWeights[0], &offsetDeltas[0], 0, &oldWeights[0], learningRate);
            }



            //add temporary weights changes to real weights (the total is divided among the total number of instances (to use the same learning rate of the standard BP)
            //it also uses momentum
            for (int w = 0; w<numOfWeights; w++) {
                float auxWeight = weights[w];
                weights[w] += (tmpWeights[w] / numOfInstances) + momentum*(auxWeight - oldWeights[w]);
                tmpWeights[w] = 0;
                oldWeights[w] = auxWeight;
            }

            if (epochs_between_reports>0 && epoch%epochs_between_reports == 0) {

                mseTrain = net->computeMSE(*trainingSet);
                if (printtype == PRINT_ALL)
                    printf("Epoch\t%d\tError on train set %.10f", epoch, mseTrain);

                if (testSet != NULL) {

                    mseTest = net->computeMSE(*testSet);
                    if (mseTest<bestMSETest) {
                        bestMSETest = mseTest;
                        if (bestMSETestNet != NULL) {
                            *bestMSETestNet = *net;
                        }
                    }
                    if ((mseTrain + mseTest)<bestMSETrainTest&&bestMSETrainTestNet != NULL) {
                        *bestMSETrainTestNet = *net;
                        bestMSETrainTest = mseTrain + mseTest;
                    }
                    if (printtype == PRINT_ALL)
                        printf("\t\tError on test set %.10f", mseTest);

                    if (bestClassTestNet != NULL) {
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

                    if (mseTest <= desired_error) {
                        if (printtype == PRINT_ALL)
                            printf("\nDesired error reached on test set.\n");
                        break;
                    }

                }

                if (printtype == PRINT_ALL)
                    printf("\n");

                if (mseTrain <= desired_error&&testSet == NULL) {
                    if (printtype == PRINT_ALL)
                        printf("Desired error reached on training set.\n");
                    break;
                }
            }
        }


        if (printtype == PRINT_ALL)
            printf("Training complete.\n");
        if (testSet != NULL) {
            return bestMSETest;
        }
        else return mseTrain;

    }

#ifdef USE_CUDA
    ///batch training on device
    ///n is the number of parameters. parameters are (float array):
    ///desired error, max_epochs, epochs_between_reports, learning_rate, momentum (using momentum is 20% slower), shuffle (SHUFFLE_ON or SHUFFLE_OFF), error function (ERROR_TANH or ERROR_LINEAR)void FeedForwardNNTrainer::trainCpuBatch(const int n, const float * params){
    float trainGPUBatch(const int n, const float * params, const int printtype) {
        //parameters parsing
        float desired_error;
        int max_epochs;
        int epochs_between_reports;
        float learningRate;
        float momentum;
        int shuff;
        int errorFunc;

        if (n<2) { printf("TOO FEW PARAMETERS SELECTED FOR TRAINING\n"); exit(1); }
        desired_error = params[0];
        max_epochs = params[1];
        if (n >= 3)
            epochs_between_reports = params[2];
        else
            epochs_between_reports = max_epochs / 10;
        if (n >= 4)
            learningRate = params[3];
        else
            learningRate = 0.7;
        if (n >= 5)
            momentum = params[4];
        else
            momentum = 0;
        if (n >= 6)
            shuff = params[5];
        else
            shuff = SHUFFLE_ON;
        if (n >= 7)
            errorFunc = params[6];
        else
            errorFunc = ERROR_TANH;

        if (printtype != PRINT_OFF) {
            printf("Training on:\t\tGPU\n");
            printf("Algorithm:\t\tBatch\n");
            printf("Desired Error:\t\t%f\n", desired_error);
            printf("Max epochs:\t\t%d\n", max_epochs);
            printf("Epochs between reports:\t%d\n", epochs_between_reports);
            printf("Learning rate:\t\t%f\n", learningRate);
            printf("Momentum:\t\t%f\n", momentum);
            if (shuff == SHUFFLE_ON)
                printf("Shuffle:\t\tON\n");
            else
                printf("Shuffle:\t\tOFF\n");
            if (errorFunc == ERROR_TANH)
                printf("Error function:\t\tTANH\n");
            else
                printf("Error function:\t\tLINEAR\n");
            printf("\n");
        }

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
        //declare some training set values
        int numOfInstances = trainingSet->getNumOfInstances();
        int numOfInputsPerInstance = trainingSet->getNumOfInputsPerInstance();
        int numOfOutputsPerInstance = trainingSet->getNumOfOutputsPerInstance();

        int numOfTestInstances = 0;

        if (testSet != NULL) {
            numOfTestInstances = testSet->getNumOfInstances();
        }

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
        float * weights;
        weights = net->getWeights();
        //declare a pointer to the net activation functions
        const int * actFuncts;
        actFuncts = net->getActFuncts();
        //declare a pointer to the net layers size
        const int * layersSize;
        layersSize = net->getLayersSize();

        //declare a pointer to the training set inputs
        float * trainingSetInputs;
        //declare a pointer to the training set outputs
        float * trainingSetOutputs;
        trainingSetInputs = trainingSet->getInputs();
        trainingSetOutputs = trainingSet->getOutputs();

        //declare a pointer to the test set inputs
        float * testSetInputs = NULL;
        //declare a pointer to the test set outputs
        float * testSetOutputs = NULL;
        if (testSet != NULL) {
            testSetInputs = testSet->getInputs();
            testSetOutputs = testSet->getOutputs();
        }

        //declare some offsets to manage array indexes of each layer 'i'
        std::vector<int> offsetWeights(numOfLayers);
        std::vector<int> offsetIns(numOfLayers);
        std::vector<int> offsetOuts(numOfLayers);
        std::vector<int> offsetDeltas(numOfLayers);

        std::vector<int> offsetTestIns(numOfLayers);
        std::vector<int> offsetTestOuts(numOfLayers);

        for (int i = 0; i<numOfLayers; i++) {
            //calculates the offsets of the arrays
            offsetWeights[i] = 0;
            offsetDeltas[i] = (layersSize[0] + 1)*numOfInstances;
            offsetIns[i] = 0;
            offsetOuts[i] = (layersSize[0] + 1)*numOfInstances;

            offsetTestIns[i] = 0;
            offsetTestOuts[i] = (layersSize[0] + 1)*numOfTestInstances;

            for (int j = 0; j<i; j++) {
                offsetWeights[i] += (layersSize[j] + 1)*layersSize[j + 1];
                offsetIns[i] += (layersSize[j] + 1)*numOfInstances;
                offsetOuts[i] += (layersSize[j + 1] + 1)*numOfInstances;
                offsetDeltas[i] += (layersSize[j + 1] + 1)*numOfInstances;

                offsetTestIns[i] += (layersSize[j] + 1)*numOfTestInstances;
                offsetTestOuts[i] += (layersSize[j + 1] + 1)*numOfTestInstances;

            }
        }

        //resets values and deltas
        for (int i = 0; i<numOfNeurons*numOfInstances; i++)values[i] = 0.0f;
        for (int i = 0; i<numOfNeurons*numOfTestInstances; i++)testValues[i] = 0.0f;
        for (int i = 0; i<numOfNeurons*numOfInstances; i++)deltas[i] = 0.0f;

        //row-major->column major indexing
        for (int i = 0; i<numOfInstances; i++) {
            for (int j = 0; j<numOfInputsPerInstance; j++)
                columnTrainingSetInputs[j*numOfInstances + i] = trainingSetInputs[i*numOfInputsPerInstance + j];
            for (int j = 0; j<numOfOutputsPerInstance; j++)
                columnTrainingSetOutputs[j*numOfInstances + i] = trainingSetOutputs[i*numOfOutputsPerInstance + j];
        }

        for (int i = 0; i<numOfTestInstances; i++) {
            for (int j = 0; j<numOfInputsPerInstance; j++)
                columnTestSetInputs[j*numOfTestInstances + i] = testSetInputs[i*numOfInputsPerInstance + j];
            for (int j = 0; j<numOfOutputsPerInstance; j++)
                columnTestSetOutputs[j*numOfTestInstances + i] = testSetOutputs[i*numOfOutputsPerInstance + j];
        }

        //copy the training set into the input neurons values
        for (int i = 0; i<numOfInstances*numOfInputsPerInstance; i++)
            values[i] = columnTrainingSetInputs[i];

        //copy the test set into the input neurons values
        for (int i = 0; i<numOfTestInstances*numOfInputsPerInstance; i++)
            testValues[i] = columnTestSetInputs[i];

        //BIAS initializations
        for (int i = 0; i<numOfLayers; i++) {
            for (int j = offsetIns[i] + (layersSize[i])*numOfInstances; j<offsetOuts[i]; j++)
                values[j] = 1.0f;
        }
        if (testSet != NULL)
            for (int i = 0; i<numOfLayers; i++) {
                for (int j = offsetTestIns[i] + (layersSize[i])*numOfTestInstances; j<offsetTestOuts[i]; j++)
                    testValues[j] = 1.0f;
            }


        //vector to shuffle training set
        std::vector<int> order(numOfInstances);
        for (int i = 0; i<numOfInstances; i++)
            order[i] = i;


        //cublas initializations
        cublasStatus stat;

        cublasInit();

        float * devValues = NULL;
        float * devTestValues = NULL;
        float * devDeltas = NULL;
        float * devWeights = NULL;
        float * devOldWeights = NULL;

        float * devTrainingSetInputs = NULL;
        float * devTrainingSetOutputs = NULL;
        float * devTestSetInputs = NULL;
        float * devTestSetOutputs = NULL;

        //allocates the vectors on the device
        stat = cublasAlloc(numOfNeurons*numOfInstances, sizeof(values[0]), (void**)&devValues);
        if (stat != CUBLAS_STATUS_SUCCESS) { printf("device memory allocation failed\n"); exit(1); }
        if (testSet != NULL) {
            stat = cublasAlloc(numOfNeurons*numOfTestInstances, sizeof(testValues[0]), (void**)&devTestValues);
            if (stat != CUBLAS_STATUS_SUCCESS) { printf("device memory allocation failed\n"); exit(1); }
        }
        stat = cublasAlloc(numOfNeurons*numOfInstances, sizeof(deltas[0]), (void**)&devDeltas);
        if (stat != CUBLAS_STATUS_SUCCESS) { printf("device memory allocation failed\n"); exit(1); }
        stat = cublasAlloc(numOfWeights, sizeof(*weights), (void**)&devWeights);
        if (stat != CUBLAS_STATUS_SUCCESS) { printf("device memory allocation failed\n"); exit(1); }
        stat = cublasAlloc(numOfWeights, sizeof(oldWeights[0]), (void**)&devOldWeights);
        if (stat != CUBLAS_STATUS_SUCCESS) { printf("device memory allocation failed\n"); exit(1); }

        stat = cublasAlloc(numOfInstances*numOfInputsPerInstance, sizeof(*devTrainingSetInputs), (void**)&devTrainingSetInputs);
        if (stat != CUBLAS_STATUS_SUCCESS) { printf("device memory allocation failed\n"); exit(1); }
        stat = cublasAlloc(numOfInstances*numOfOutputsPerInstance, sizeof(*devTrainingSetOutputs), (void**)&devTrainingSetOutputs);
        if (stat != CUBLAS_STATUS_SUCCESS) { printf("device memory allocation failed\n"); exit(1); }
        if (testSet != NULL) {
            stat = cublasAlloc(numOfTestInstances*numOfInputsPerInstance, sizeof(*devTestSetInputs), (void**)&devTestSetInputs);
            if (stat != CUBLAS_STATUS_SUCCESS) { printf("device memory allocation failed\n"); exit(1); }
            stat = cublasAlloc(numOfTestInstances*numOfOutputsPerInstance, sizeof(*devTestSetOutputs), (void**)&devTestSetOutputs);
            if (stat != CUBLAS_STATUS_SUCCESS) { printf("device memory allocation failed\n"); exit(1); }
        }

        //copies the training set inputs and outputs on the device
        cudaMemcpy(devTrainingSetInputs, &columnTrainingSetInputs[0], numOfInstances*numOfInputsPerInstance * sizeof(columnTrainingSetInputs[0]), cudaMemcpyHostToDevice);
        cudaMemcpy(devTrainingSetOutputs, &columnTrainingSetOutputs[0], numOfInstances*numOfOutputsPerInstance * sizeof(columnTrainingSetOutputs[0]), cudaMemcpyHostToDevice);

        if (testSet != NULL) {
            //copies the test set inputs and outputs on the device
            cudaMemcpy(devTestSetInputs, &columnTestSetInputs[0], numOfTestInstances*numOfInputsPerInstance * sizeof(columnTestSetInputs[0]), cudaMemcpyHostToDevice);
            cudaMemcpy(devTestSetOutputs, &columnTestSetOutputs[0], numOfTestInstances*numOfOutputsPerInstance * sizeof(columnTestSetOutputs[0]), cudaMemcpyHostToDevice);
        }

        //copies the training set inputs with the biases and the weights to the device
        cudaMemcpy(devValues, &values[0], numOfNeurons*numOfInstances * sizeof(values[0]), cudaMemcpyHostToDevice);

        if (testSet != NULL) {
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
            if (testSet != NULL) {
                printf("\t\tError on test set %.10f", GPUComputeMSE(devTestValues, devWeights, actFuncts, numOfLayers, layersSize, numOfTestInstances, numOfOutputsPerInstance, devTestSetOutputs, &offsetTestIns[0], &offsetWeights[0], &offsetTestOuts[0]));
            }
            printf("\n");
        }

        //epochs training
        for (int epoch = 1; epoch <= max_epochs&&quit == false; epoch++) {

            //shuffle instances
            int ind = 0, aux = 0;
            if (shuff == SHUFFLE_ON)
                for (int i = 0; i<numOfInstances; i++) {
                    ind = gen_random_int(i, numOfInstances - 1);
                    aux = order[ind];
                    order[ind] = order[i];
                    order[i] = aux;
                }

            //TESTING  MINI BATCHES
            /*
            //training
            int minibdim = 1000000;
            for(int minib = 0; minib < numOfInstances; minib += minibdim) {
            int num_mini_instances = minibdim;
            if (minib + minibdim > numOfInstances)
            num_mini_instances = numOfInstances - minib;

            std::vector<int> oins = offsetIns;
            int lay = 0;
            for (auto && el : oins) {
            el += minib;// *(layersSize[lay] + 1);/// column major
            ++lay;
            }

            lay = 0;
            std::vector<int> oouts = offsetOuts;
            for (auto && el : oouts) {
            // if (lay + 1 < numOfLayers)
            el += minib;// * (layersSize[lay + 1] + 1);/// dimensione di ciascun layer
            //else
            //  el += (layersSize[lay]) * minib;/// dimensione di ciascun layer
            ++lay;
            }

            lay = 0;
            std::vector<int> odeltas = offsetDeltas;
            for (auto && el : odeltas) {
            //  if (lay + 1 < numOfLayers)
            el += minib;// *(layersSize[lay + 1] + 1);/// dimensione di ciascun layer
            // else
            //   el += (layersSize[lay]) * minib;/// dimensione di ciascun layer
            ++lay;
            }


            GPUForward(devValues, devWeights,
            actFuncts, numOfLayers, layersSize, numOfInstances, num_mini_instances,
            &oins[0], &offsetWeights[0], &oouts[0]);

            //computes all the instances backward of the backpropagation training
            GPUBack(devValues, devWeights, devDeltas,
            actFuncts, numOfLayers, layersSize, numOfInstances, num_mini_instances, numOfOutputsPerInstance,
            devTrainingSetOutputs + minib, &offsetWeights[0], &odeltas[0], &oouts[0],
            errorFunc);

            //update the weights using the deltas
            GPUUpdate(devValues, devWeights, devDeltas,
            numOfLayers, layersSize, numOfInstances, num_mini_instances,
            &oins[0], &offsetWeights[0], &odeltas[0],
            momentum, devOldWeights, learningRate);

            }
            */

            GPUForward(devValues, devWeights,
                actFuncts, numOfLayers, layersSize, numOfInstances, numOfInstances,
                &offsetIns[0], &offsetWeights[0], &offsetOuts[0]);

            //computes all the instances backward of the backpropagation training
            GPUBack(devValues, devWeights, devDeltas,
                actFuncts, numOfLayers, layersSize, numOfInstances, numOfInstances, numOfOutputsPerInstance,
                devTrainingSetOutputs, &offsetWeights[0], &offsetDeltas[0], &offsetOuts[0],
                errorFunc);

            //update the weights using the deltas
            GPUUpdate(devValues, devWeights, devDeltas,
                numOfLayers, layersSize, numOfInstances, numOfInstances,
                &offsetIns[0], &offsetWeights[0], &offsetDeltas[0],
                momentum, devOldWeights, learningRate);

            if (epochs_between_reports>0 && epoch%epochs_between_reports == 0) {

                cudaMemcpy(weights, devWeights, numOfWeights * sizeof(float), cudaMemcpyDeviceToHost);

                //float mseTrain=net->computeMSE(*trainingSet);
                mseTrain = GPUComputeMSE(devValues, devWeights, actFuncts, numOfLayers, layersSize, numOfInstances, numOfOutputsPerInstance, devTrainingSetOutputs, &offsetIns[0], &offsetWeights[0], &offsetOuts[0]);
                if (printtype == PRINT_ALL)
                    printf("Epoch    %d    Error on train set %.10f", epoch, mseTrain);

                if (testSet != NULL) {

                    //float mseTest=net->computeMSE(*testSet);
                    mseTest = GPUComputeMSE(devTestValues, devWeights, actFuncts, numOfLayers, layersSize, numOfTestInstances, numOfOutputsPerInstance, devTestSetOutputs, &offsetTestIns[0], &offsetWeights[0], &offsetTestOuts[0]);
                    if (mseTest<bestMSETest) {
                        bestMSETest = mseTest;
                        if (bestMSETestNet != NULL) {
                            *bestMSETestNet = *net;
                        }
                    }
                    if ((mseTrain + mseTest)<bestMSETrainTest&&bestMSETrainTestNet != NULL) {
                        *bestMSETrainTestNet = *net;
                        bestMSETrainTest = mseTrain + mseTest;
                    }
                    if (printtype == PRINT_ALL)
                        printf("        Error on test set %.10f", mseTest);

                    if (bestClassTestNet != NULL) {
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

                    if (mseTest <= desired_error) {
                        if (printtype == PRINT_ALL)
                            printf("\nDesired error reached on test set.\n");
                        break;
                    }

                }

                if (printtype == PRINT_ALL)
                    printf("\n");

                if (mseTrain <= desired_error&&testSet == NULL) {
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
        if (testSet != NULL) {
            return bestMSETest;
        }
        else return mseTrain;

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

#ifdef USE_CUDA
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
