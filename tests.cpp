/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

#define USE_CUDA

#include <stdio.h>
#include <iostream>

#include <math.h>
#include <stdlib.h>

#include <vector>

//#include "genetics/GAFeedForwardNN.h"
#include "FeedForwardNN.h"
#include "LearningSet.h"
#include "FeedForwardNNTrainer.h"
#include "ActivationFunctions.h"

using namespace std;

void retest(const FeedForwardNN & mynet, const LearningSet & testSet) {

    int good = 0;
    for (int n = 0; n < testSet.getNumOfInstances(); ++n) {


        auto in = testSet.get_input_n(n);
        auto out = testSet.get_output_n(n);

        std::vector<float> res(testSet.getNumOfOutputsPerInstance());
        mynet.compute(in, &res[0]);

        int best_out_ind = 0;
        float best_out = 0;
        int best_res_ind = 0;
        float best_res = 0;
        for (int i = 0; i < testSet.getNumOfOutputsPerInstance(); ++i) {
           // std::cout << "Exp: " << out[i] << " actual: " << res[i] << "\n";
            if (out[i] > best_out) {
                best_out = out[i];
                best_out_ind = i;
            }
            if (res[i] > best_res) {
                best_res = res[i];
                best_res_ind = i;
            }
        }

        if (best_res_ind == best_out_ind) {
            ++good;
        }
    }
    std::cout << "Fraction: " << 1.0 * good / testSet.getNumOfInstances() << "\n";

}

int main(){

	//TRAINING EXAMPLE

	//LearningSet trainingSet(R"(C:\Users\Luca\Desktop\libcudann.build\mushroom.train)");
	//LearningSet testSet(R"(C:\Users\Luca\Desktop\libcudann.build\mushroom.test)");

    LearningSet trainingSet(R"(C:\Users\Luca\Desktop\adidas_project\trunk\vision_code\feature_extraction.build\train.set)");
    LearningSet testSet(R"(C:\Users\Luca\Desktop\adidas_project\trunk\vision_code\feature_extraction.build\train.set)");

	//layer sizes
	//int layers[]={125,100,2};
    //int functs[] = { 3,3,1 };
    int layers[] = { 200 * 200 * 3,1000, 500, 500,500,200,100,4 };
	//activation functions (1=sigm,2=tanh)
	int functs[]={3,2,3,2,3,2,3,1};
	//declare the network with the number of layers
	FeedForwardNN mynet(8,layers,functs);
	
	FeedForwardNNTrainer trainer;
	trainer.selectNet(mynet);
	trainer.selectTrainingSet(trainingSet);
	trainer.selectTestSet(testSet);

	//optionally save best net found on test set, or on train+test, or best classifier
	//FeedForwardNN mseT;
	//FeedForwardNN mseTT;
	FeedForwardNN cl;
	//trainer.selectBestMSETestNet(mseT);
	//trainer.selectBestMSETrainTestNet(mseTT);
	trainer.selectBestClassTestNet(cl);

	//parameters:
	//TRAIN_GPU - TRAIN_CPU
	//ALG_BATCH - ALG_BP (batch packpropagation or standard)
	//desired error
	//total epochs
	//epochs between reports
	//learning rate
	//momentum
	//SHUFFLE_ON - SHUFFLE_OFF
	//error computation ERROR_LINEAR - ERROR_TANH
	
	float param[]={TRAIN_GPU,ALG_BATCH,0.00,10000,10,0.01,0.0,SHUFFLE_ON,ERROR_LINEAR};
    //float param[] = { TRAIN_CPU,ALG_BP,0.00,20,4,0.1,0,SHUFFLE_ON,ERROR_TANH };
	trainer.train(9,param);
	 
	
	mynet.saveToTxt("../mynetmushrooms.net");
	
	//mseT.saveToTxt("../mseTmushrooms.net");
	//mseTT.saveToTxt("../mseTTmushrooms.net");
	//cl.saveToTxt("../clmushrooms.net");


    retest(mynet, testSet);

    getchar();

/*	//EVOLUTION EXAMPLE

	LearningSet trainingSet("../bcw.train");
	LearningSet testSet("../bcw.test");

	GAFeedForwardNN evo;

	//choose a net to save the best training
	FeedForwardNN mybest;
	evo.selectBestNet(mybest);

	evo.selectTrainingSet(trainingSet);
	evo.selectTestSet(testSet);

	//evolution parameters:
	//popolation
	//generations
	//selection algorithm ROULETTE_WHEEL - TOURNAMENT_SELECTION
	//training for each generated network
	//crossover probability
	//mutation probability
	//number of layers
	//max layer size
	evo.init(30,100,ROULETTE_WHEEL,2,0.5,0.3,2,700);

	//training parameters:
	//TRAIN_GPU - TRAIN_CPU
	//ALG_BATCH - ALG_BP (batch packpropagation or standard)
	//desired error
	//total epochs
	//epochs between reports

	float param[]={TRAIN_GPU,ALG_BATCH,0.00,800,1};

	evo.evolve(5,param,PRINT_MIN);

	mybest.saveToTxt("../mybestbwc.net");
*/

/*	//USAGE EXAMPLE
	//load a trained network from a file
	FeedForwardNN net("mytrainedxornet.net");
	float in[2],out[1];
	in[0]=1;
	in[1]=0;
	
	//compute the network (for example an xor function) from inputs in[0] and in[1] and puts the result in out[0]
	net.compute(in,out);
	printf("%f\n",out[0]);
*/
}




