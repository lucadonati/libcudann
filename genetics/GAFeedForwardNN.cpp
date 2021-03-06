/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

/*
 * GAFeedForwardNN.cpp
 *
 *  Created on: Jan 20, 2011
 *      Author: donati
 */

#include "GAFeedForwardNN.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#include "../RandomGenerator.h"

#define GAUSSIAN_LIM 3.0f

GAFeedForwardNN::GAFeedForwardNN() {
	trainingSet = NULL;
	testSet = NULL;
	net = NULL;
	bestNet = NULL;


	chromosomes=NULL;


	popsize=0;
	generations=0;
	numberofevaluations=0;
	selectionalgorithm=ROULETTE_WHEEL;
	pcross=0;
	pmut=0;
	nhidlayers=0;
	maxdimhiddenlayers=0;
}

GAFeedForwardNN::~GAFeedForwardNN() {
	delete []chromosomes;
}

//choose the training set
void GAFeedForwardNN::selectTrainingSet(LearningSet & s)
{
	trainingSet = &s;
}

//choose the test set. if this is set the error rate is computed on test set instead of training set
void GAFeedForwardNN::selectTestSet(LearningSet & s)
{
	testSet = &s;
}

//choose a net to save the best network individual trained so far. mse on test set (or train if no test is specified) is the criterion
void GAFeedForwardNN::selectBestNet(FeedForwardNN & n)
{
	bestNet = &n;
}

//initialize the genetic algorithm parameters and create the first population of individuals
void GAFeedForwardNN::init(const int pop,const int gen,const int selection,const int neval,const float pc,const float pm,const int nhid,const int maxdimhid){

	popsize=pop;
	generations=gen;
	selectionalgorithm=selection;
	numberofevaluations=neval;
	pcross=pc;
	pmut=pm;
	nhidlayers=nhid;
	maxdimhiddenlayers=maxdimhid;

	if(popsize<=1){throw std::runtime_error("POPULATION SIZE SHOULD BE 2 OR MORE");}
	if(generations<1){ throw std::runtime_error("AT LEAST ONE GENERATION SHOULD BE EVOLVED");}
	if(selectionalgorithm<0||selectionalgorithm>2){ throw std::runtime_error("SELECTION ALGORITHM NOT IMPLEMENTED YET");}
	if(numberofevaluations<1){ throw std::runtime_error("AT LEAST ONE EVALUATION SHOULD BE DONE");}
	if(pcross<0||pcross>1){ throw std::runtime_error("CROSSOVER PROBABILITY SHOULD BE BETWEEN 0 AND 1");}
	if(pmut<0||pmut>1){ throw std::runtime_error("MUTATION PROBABILITY SHOULD BE BETWEEN 0 AND 1");}
	if(nhidlayers<0){ throw std::runtime_error("HIDDEN LAYERS NUMBER CAN'T BE NEGATIVE");}
	if(maxdimhiddenlayers<1){ throw std::runtime_error("HIDDEN LAYERS SHOULD HAVE AT LEAT 1 NEURON");}

	delete [] chromosomes;
	chromosomes=new FloatChromosome[popsize];

	//generates the population
	for(int i=0;i<popsize;i++){
		int chromoLenght=nhidlayers+nhidlayers+2+1;
		chromosomes[i].setSize(chromoLenght);
		//each value is set randomly
		for(int j=0;j<chromoLenght;j++){
			float min,max,value;
			//dimension of each hidden layer (random int between 1 and maxdimhiddenlayers)
			if(j<nhidlayers){
				min=1;
				max=maxdimhiddenlayers;
                value = gen_random_int(min, max);
			}
			//activation function of each hidden layer (1 for sigmoid, 2 for tanh, random)
			else if(j<nhidlayers+nhidlayers+2){
				min=1;
				max=2;
                value = gen_random_int(min, max);
			}
			//learning rate (random between 0 and 1)
			else{
				min=0;
				max=1;
                value = gen_random_real(min, max);
			}
			chromosomes[i].setElement(j,value);
		}
	}


}
//run the genetic algorithm initialized before with some training parameters:
//training location, training algorithm, desired error, max_epochs, epochs_between_reports
//see "FeedForwardNNTrainer" class for more details
//printtype specifies how much verbose will be the execution (PRINT_ALL,PRINT_MIN,PRINT_OFF)
void GAFeedForwardNN::evolve(const int n, const float * params, const int printtype){

	if(n<5){throw std::runtime_error("TOO FEW PARAMETERS FOR TRAINING");}
    std::vector<int> layers(nhidlayers+2);
    std::vector<int> functs(nhidlayers+2);
	float learningRate;

	std::vector<float> fitnesses(popsize);
	float totfitness=0;
	float bestFitnessEver=0;
    std::vector<FloatChromosome> newpop(popsize);

	layers[0]=trainingSet->getNumOfInputsPerInstance();
	layers[nhidlayers+1]=trainingSet->getNumOfOutputsPerInstance();

	//for each generation
	for(int gen=0;gen<generations;gen++){
		float bestFitnessGeneration=0;
		int bestFitGenIndex=0;
		totfitness=0;

		std::cout << "GENERATION NUMBER:\t" << gen << "\n\n";

		//fitness evaluation of each individual
		for(int i=0;i<popsize;i++){

			std::cout << "\nINDIVIDUAL N:\t" << i << "\n";

			//decode the chromosome hidden layers sizes
			for(int j=0;j<nhidlayers;j++){
				layers[j+1]=chromosomes[i].getElement(j);
			}
			//decode the chromosome activation functions for each layer
			for(int j=0;j<nhidlayers+2;j++){
				functs[j]=chromosomes[i].getElement(j+nhidlayers);
			}
			//decode the chromosome learning rate
			learningRate=chromosomes[i].getElement(nhidlayers+nhidlayers+2);

			float medium=0;

			FeedForwardNN mseT;

			//do a number of evaluations with different weights and average the results
			for(int n=0;n<numberofevaluations;n++){

				//choose what to print based on user's choice
				int print=PRINT_ALL;
				if(printtype==PRINT_MIN){
					if(n==0)
						print=PRINT_MIN;
					else
						print=PRINT_OFF;
				}
				if(printtype==PRINT_OFF)
					print=PRINT_OFF;

				//decode the chromosome into a real network
				FeedForwardNN net(nhidlayers+2,&layers[0],&functs[0]);

				FeedForwardNNTrainer trainer;
				trainer.selectTrainingSet(*trainingSet);
				if(testSet!=NULL){
					trainer.selectTestSet(*testSet);
				}
				trainer.selectNet(net);

				trainer.selectBestMSETestNet(mseT);

				float par[]={params[0],params[1],params[2],params[3],params[4],learningRate,0,SHUFFLE_ON,ERROR_TANH};

				//do the training of the net and evaluate is MSE error
				medium+=trainer.train(9,par,print)/float(numberofevaluations);
			}

			//the fitness is computed as the inverse of the MSE
			fitnesses[i]=1.0f/medium;

			printf("FITNESS:\t%.2f\n\n",fitnesses[i]);

			//updates the best individual of the generation
			if(fitnesses[i]>bestFitnessGeneration){bestFitnessGeneration=fitnesses[i];bestFitGenIndex=i;}

			//if this is the best fitness ever it store the network in bestNet
			if(bestNet!=NULL)
			if(fitnesses[i]>bestFitnessEver){*bestNet=mseT;bestFitnessEver=fitnesses[i];}

			totfitness+=fitnesses[i];
		}

		//the best individual is always carried to the next generation
		newpop[0]=chromosomes[bestFitGenIndex];

		//generate the new population
		for(int i=1;i<popsize;i++){
			//selection
			int firstmate=0,secondmate=0;

			//first mate
			switch(selectionalgorithm){
				case ROULETTE_WHEEL:		firstmate=rouletteWheel(popsize,&fitnesses[0]);					break;
				case TOURNAMENT_SELECTION:	firstmate=tournament(popsize,&fitnesses[0],popsize/5+1);		break;
                default:					throw std::runtime_error("SELECTION ALGORITHM NOT IMPLEMENTED YET");break;
			}
			//second mate
			do{
				switch(selectionalgorithm){
					case ROULETTE_WHEEL:		secondmate=rouletteWheel(popsize,&fitnesses[0]);				break;
					case TOURNAMENT_SELECTION:	secondmate=tournament(popsize,&fitnesses[0],popsize/5+1);		break;
					default:					throw std::runtime_error("SELECTION ALGORITHM NOT IMPLEMENTED YET");break;
				}
			}while(firstmate==secondmate);


			FloatChromosome child;
			//do the crossover
			child=crossover(chromosomes[firstmate],chromosomes[secondmate],pcross);
			//and the mutation
			child=mutation(child,pmut,maxdimhiddenlayers,nhidlayers);
			//and put the child in the new generation
			newpop[i]=child;
		}

		//copy the new generation over the older one, wich is the one we will still use
		for(int i=0;i<popsize;i++){
			chromosomes[i]=newpop[i];
		}
	}

}

//performs crossover between a chromosome and a mate, and return the result
FloatChromosome crossover(const FloatChromosome & first, const FloatChromosome & second, const float pcross){
	float roll;

	FloatChromosome ret=first;
	FloatChromosome app=second;

	for(int i=0;i<ret.getSize();i++){
        roll = gen_random_real(0, 1);
		if(roll<pcross){
			ret.setElement(i,app.getElement(i));
		}
	}

	return ret;
}

//performs mutation on a chromosome, and return the result
FloatChromosome mutation(const FloatChromosome & first, const float pmut, const int maxdimhiddenlayers,const int nhidlayers){
	float roll;
	FloatChromosome ret=first;
	for(int i=0;i<ret.getSize();i++){
        roll = gen_random_real(0, 1);
		if(roll<pmut){
			if(i<nhidlayers){
				float x1,x2,y1;
				x1= gen_random_real(0, 1);
				x2= gen_random_real(0, 1);
				y1 = sqrt(-2*log(x1))*cos(2*M_PI*x2);
				if(y1>GAUSSIAN_LIM)y1=GAUSSIAN_LIM;
				if(y1<-GAUSSIAN_LIM)y1=-GAUSSIAN_LIM;
	//TODO attenzione non converga sui bordi
				ret.setElement(i,(int)(ret.getElement(i)+y1*(float)maxdimhiddenlayers/(2.0f*GAUSSIAN_LIM)));
				float val=ret.getElement(i);
				if(val<1)ret.setElement(i,1.0f);
				if(val>maxdimhiddenlayers)ret.setElement(i,maxdimhiddenlayers);
			}
			else if(i<nhidlayers+nhidlayers+2){
				if((int)(ret.getElement(i))==1)
					ret.setElement(i,2.0f);
				else
					ret.setElement(i,1.0f);
			}
			else{
				float x1,x2,y1;
				x1= gen_random_real(0, 1);
				x2= gen_random_real(0, 1);
				y1 = sqrt(-2*log(x1))*cos(2*M_PI*x2);
				if(y1>GAUSSIAN_LIM)y1=GAUSSIAN_LIM;
				if(y1<-GAUSSIAN_LIM)y1=-GAUSSIAN_LIM;
	//TODO attenzione non converga sui bordi
				ret.setElement(i,(ret.getElement(i)+y1/(2.0f*GAUSSIAN_LIM)));
				float val=ret.getElement(i);
				if(val<0.0f)ret.setElement(i,0.0f);
				if(val>1.0f)ret.setElement(i,1.0f);
			}
		}
	}
	return ret;
}

//roulette wheel selection
int rouletteWheel(const int size, const float * fitnesses){
	//computes the total fitness
	float totfitness=0;
	for(int i=0;i<size;i++){
		totfitness+=fitnesses[i];
	}

	//spin the ball between 0 and that total fitness
	float spin=totfitness*gen_random_real(0, 1);

	int chosen=0;
	//pick the relative individual
	for(chosen=0;chosen<size;chosen++){
		spin-=fitnesses[chosen];
		if(spin<0)break;
	}

	//and returns it
	return chosen;
}

//tournament selection
int tournament(const int size, const float * fitnesses, const int toursize){

	std::vector<int> picks(toursize);

	float bestfitness=0;
	int bestindex=0;

	//fills the tournament
	for(int i=0;i<toursize;i++){
		int pick;
		bool alreadyPicked=false;
		do{
			//with random individuals
			alreadyPicked=false;
			pick= gen_random_int(0, size - 1);

			for(int j=0;j<i;j++){
				if(picks[j]==pick){
					alreadyPicked=true;
					break;
				}
			}

		//checking they are picked at most only once each
		}while(alreadyPicked==true);
		picks[i]=pick;

		//the individual with the best fitness wins the tournament
		if(fitnesses[i]>bestfitness){
			bestfitness=fitnesses[i];
			bestindex=picks[i];
		}
	}

	//and is selected
	return bestindex;
}
