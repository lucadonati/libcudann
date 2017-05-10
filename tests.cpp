/*
libcudann
Copyright (C) 2011 Luca Donati (lucadonati85@gmail.com)
*/

//#define DISABLE_CUDA_NN

#include <cstdio>
#include <iostream>

#include <cmath>
#include <cstdlib>

#include <vector>
#include <list>

//#include "genetics/GAFeedForwardNN.h"
#include "FeedForwardNN.h"
#include "LearningSet.h"
#include "FeedForwardNNTrainer.h"
#include "ActivationFunctions.h"

using namespace std;

inline void test_classification(const FeedForwardNN & mynet, const LearningSet & testSet) {

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

void test_all(const FeedForwardNN & mynet, const LearningSet & testSet) {

    for (int n = 0; n < testSet.getNumOfInstances(); ++n) {


        auto in = testSet.get_input_n(n);
        auto out = testSet.get_output_n(n);

        std::vector<float> res(testSet.getNumOfOutputsPerInstance());
        mynet.compute(in, &res[0]);

        for (int i = 0; i < testSet.getNumOfOutputsPerInstance(); ++i) {
            std::cout << "Exp: " << out[i] << " actual: " << res[i] << "\n";
            
        }

    }

}

int main(){
    /*
    //TRAINING EXAMPLE
    std::string base = R"(C:\Users\Luca\Desktop\adidas_project\trunk\vision_code\feature_extraction.build\)";

    std::vector<LearningSet> sets;
    sets.push_back(LearningSet::readSimplifiedSet(std::string(base + "bigset-zero.set").c_str()));
    sets.push_back(LearningSet::readSimplifiedSet(std::string(base + "bigset-one.set").c_str()));
    sets.push_back(LearningSet::readSimplifiedSet(std::string(base + "bigset-two.set").c_str()));
    sets.push_back(LearningSet::readSimplifiedSet(std::string(base + "bigset-three.set").c_str()));

    int n = 0;
    auto trei = [&] (auto&&, auto &&, auto &&) {return n++ < 850; };
    auto tes = [&](auto&&, auto &&, auto &&) {return !trei(0, nullptr, nullptr); };

    LearningSet tr, te;

    for (int i = 0; i < 4; ++i) {
        n = 0;
        auto tre = sets[i].filter_set_by(trei);
        tr.insert(tre);

        n = 0;
        auto tess = sets[i].filter_set_by(tes);
        te.insert(tess);
    }
    tr.writeBinarySet("adi_train.set");
    te.writeBinarySet("adi_test.set");
    
    
    std::cout << "ok" << "\n";
    int s;
    std::cin >> s;
    
    LearningSet trainingSet;

    trainingSet = trainingSet.shuffle();

    int tot = 0;
    auto testSet = trainingSet.filter_set_by([&](auto && n, auto && i, auto && o) {return tot++ > 4000; });
    tot = 0;
    trainingSet = trainingSet.filter_set_by([&](auto && n, auto && i, auto && o) {return tot++ < 4000; });
    */
    //std::vector<int> layers = { 13,300,200,1 };
    //std::vector<int> functs = { ACT_RELU, ACT_RELU, ACT_RELU, ACT_SIGMOID };

    //LearningSet trainingSet(R"(C:\Users\Luca\Desktop\cuda-libcuda\xor.train)");
    //LearningSet testSet(R"(C:\Users\Luca\Desktop\cuda-libcuda\xor.train)");
    //std::vector<int> layers = { 2,8,4,1 };
    //std::vector<int> functs = { ACT_RELU, ACT_RELU, ACT_RELU, ACT_SIGMOID };


    //LearningSet trainingSet(R"(C:\Users\Luca\Desktop\libcudann.build\mushroom.train)");
    //LearningSet testSet(R"(C:\Users\Luca\Desktop\libcudann.build\mushroom.test)");
    //std::vector<int> layers={125,100,2};
    //std::vector<int> functs = { 3,3,1 };

    LearningSet trainingSet = LearningSet::readBinarySet("adi_train.set");
    int n = 0;
   // trainingSet = trainingSet.shuffle().filter_set_by([&](auto&&, auto &&, auto &&) {return n++ < 400 * 4; });
    trainingSet = trainingSet.shuffle();
    LearningSet testSet = LearningSet::readBinarySet("adi_test.set");
    std::vector<LearningSet> batches;
    for (int i = 0; i < trainingSet.getNumOfInstances(); i += 200 * 3) {
        n = 0;
        batches.push_back(trainingSet.filter_set_by([&] (auto,auto,auto){++n; return n > i && n < i + 600; }));
    }
    //LearningSet trainingSet(R"(C:\Users\Luca\Desktop\adidas_project\trunk\vision_code\feature_extraction.build\train.set)");
    //LearningSet testSet(R"(C:\Users\Luca\Desktop\adidas_project\trunk\vision_code\feature_extraction.build\train.set)");
    std::vector<int> layers = { 200 * 200 * 3, 1000, 800,800,500,4 };
    std::vector<int> functs={3,3,3,3,3,1};

    //layer sizes
    //activation functions (1=sigm,2=tanh,3=relu)

    
    //declare the network with the number of layers
    //FeedForwardNN mynet(3, layers, functs);
    FeedForwardNN mynet(layers.size(),&layers[0],&functs[0]);
    mynet.initWeights(-0.01, 0.01);
    //mynet.initWidrowNguyen(testSet);
    
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
    for(int i=0;i<100;++i)
        for (auto&& tr1 : batches) {
            trainer.selectTrainingSet(tr1);
            // mynet.initWidrowNguyen(trainingSet);
                
            TrainingParameters params;
            params.training_location = TRAIN_GPU;
            params.training_algorithm = ALG_BATCH;
            params.desired_error = 0.0;
            params.max_epochs = 30;
            params.epochs_between_reports = 30;
            params.learningRate = 0.001;
            params.momentum = 0.2;
            params.shuff = SHUFFLE_ON;
            params.errorFunc = ERROR_LINEAR;
        
            trainer.train(params);
        }
    
    mynet.saveToTxt("mynetmushrooms.net");
    
    //mseT.saveToTxt("../mseTmushrooms.net");
    //mseTT.saveToTxt("../mseTTmushrooms.net");
    cl.saveToTxt("clmushrooms.net");
    std::cout << "saved" << "\n";

    test_classification(cl, testSet);
    test_all(cl, testSet);

    getchar();

/*    //EVOLUTION EXAMPLE

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

/*    //USAGE EXAMPLE
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




