/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/
#define DISABLE_CUDA_NN
#include <stdio.h>
#include <iostream>

#include <math.h>
#include <stdlib.h>

#include <genetics/GAFeedForwardNN.h>
#include <FeedForwardNN.h>
#include <LearningSet.h>
#include <FeedForwardNNTrainer.h>
#include <ActivationFunctions.h>

using namespace std;



int main(){

	//TRAINING EXAMPLE

    auto trainingSet = LearningSet::readFannSet ("mushroom.train");
	auto testSet = LearningSet::readFannSet("mushroom.test");

	//layer sizes
	std::vector<int> layers = {125,30,2};
	//activation functions (1=sigm,2=tanh)
    std::vector<int> functs = {2,1,2};
	//declare the network with the number of layers
	FeedForwardNN mynet(layers,functs);
	
	FeedForwardNNTrainer trainer;
	trainer.selectNet(mynet);
	trainer.selectTrainingSet(trainingSet);
	trainer.selectTestSet(testSet);

	//optionally save best net found on test set, or on train+test, or best classifier
	//FeedForwardNN mseT;
	//FeedForwardNN mseTT;
	//FeedForwardNN cl;
	//trainer.selectBestMSETestNet(mseT);
	//trainer.selectBestMSETrainTestNet(mseTT);
	//trainer.selectBestClassTestNet(cl);

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
		
    TrainingParameters params;
    params.training_location = TRAIN_CPU;
    params.training_algorithm = ALG_BATCH;
    params.desired_error = 0.0;
    params.max_epochs = 1000;
    params.epochs_between_reports = 10;
    params.learningRate = 0.1;
    params.momentum = 0.0;
    params.shuff = SHUFFLE_ON;
    params.errorFunc = ERROR_LINEAR;
	trainer.train(params);
	 
	
	mynet.saveToTxt("../mynetmushrooms.net");
	
	//mseT.saveToTxt("../mseTmushrooms.net");
	//mseTT.saveToTxt("../mseTTmushrooms.net");
	//cl.saveToTxt("../clmushrooms.net");


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
	std::cout << out[0] << "\n";
*/
}




