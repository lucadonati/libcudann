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

#include "Plot.h"


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

int test_1() {
    {
        std::string trai = R"(
        1 2 1
        1 0
        1
        )";
        std::ofstream tt("tt.fann");
        tt << trai;
    }
    
        LearningSet trainingSet = LearningSet::readFannSet("tt.fann");
        std::vector<int> layers = { 2, 1 };
        std::vector<int> functs = { 3, 0 };

        FeedForwardNN mynet(layers.size(), &layers[0], &functs[0]);

        auto * wei = mynet.getWeights();
        int num = mynet.getNumOfWeights();
        wei[0] = 0.1;
        wei[1] = -0.3;
        wei[2] = 0.0;
        

        FeedForwardNNTrainer trainer;
        trainer.selectNet(mynet);
        trainer.selectTrainingSet(trainingSet);
        

        TrainingParameters params;
        params.training_location = TRAIN_CPU;
        params.training_algorithm = ALG_BP;
        params.desired_error = 0.0;
        params.max_epochs = 150;
        params.epochs_between_reports = 1;
        params.learningRate = 0.1;
        params.momentum = 0.0;
        params.shuff = SHUFFLE_ON;
        params.errorFunc = ERROR_LINEAR;

        trainer.train(params);


        getchar();

        return 0;
}

int test_2(){
    
    /*
    auto aditrain = LearningSet::readBinarySet("adi_train.set");
    auto aditest = LearningSet::readBinarySet("adi_test.set");

    aditrain.writeInOutSet("adi_train_ins", "adi_train_outs");
    aditest.writeInOutSet("adi_test_ins", "adi_test_outs");
    return 0;


    std::string base = R"(c:\users\luca\desktop\adidas_project\trunk\vision_code\my_tom_tests_adi2\)";

    
    auto fronti = LearningSet::readSimplifiedSet(std::string(base +"points_front.set").c_str());
    auto retri = LearningSet::readSimplifiedSet(std::string(base + "points_back.set").c_str());

 //   fronti = fronti.shuffle();
 //   retri = retri.shuffle();

    int split = 15 * 20;

    int n = 0;
    n = 0;
    auto testfronti = fronti.filter_set_by([&](auto, auto, auto) {
        return n++ < split;
    });
    n = 0;    
    auto trainfronti = fronti.filter_set_by([&](auto, auto, auto) {
        if (n >= split && n < 27 * 20) {
            ++n;
            return true;
        }
        ++n;
        return false;
    });

    n = 0;
    auto testretri = retri.filter_set_by([&](auto, auto, auto) {
        return n++ < split;
    });
    n = 0;
    auto trainretri = retri.filter_set_by([&](auto, auto, auto) {
        if (n >= split && n < 27 * 20) {
            ++n;
            return true;
        }
        ++n;
        return false;
    });

    auto alltrain = trainfronti;
    alltrain.insert(trainretri);

    auto alltest = testfronti;
    alltest.insert(testretri);

    alltrain = alltrain.shuffle();
    alltrain.writeFannSet(std::string(base + "tom_train.set").c_str());
    alltest = alltest.shuffle();
    alltest.writeFannSet(std::string(base + "tom_test.set").c_str());
    */
    /*
    {
        LearningSet set;
        //std::ifstream ifs_i(R"(c:\users\luca\desktop\train-images.idx3-ubyte)", std::ofstream::binary);
        //std::ifstream ifs_o(R"(c:\users\luca\desktop\train-labels.idx1-ubyte)", std::ofstream::binary);

        //std::ofstream ofs(R"(c:\users\luca\desktop\minst.simp_train)");

        std::ifstream ifs_i(R"(c:\users\luca\desktop\t10k-images.idx3-ubyte)", std::ofstream::binary);
        std::ifstream ifs_o(R"(c:\users\luca\desktop\t10k-labels.idx1-ubyte)", std::ofstream::binary);

        std::ofstream ofs(R"(c:\users\luca\desktop\minst.simp_test)");

        auto bin_read = [](auto && s, auto && t) {
            s.read(reinterpret_cast<char*>(&t), sizeof(t));
        };

        bin_read(ifs_i, int32_t());
        bin_read(ifs_i, int32_t());
        bin_read(ifs_i, int32_t());
        bin_read(ifs_i, int32_t());

        bin_read(ifs_o, int32_t());
        bin_read(ifs_o, int32_t());

        std::vector<uint8_t> data;

        while (true) {
            data.resize(28 * 28);
            for (int i = 0; i < 28 * 28; ++i) {
                bin_read(ifs_i, data[i]);
            }
            if (!ifs_i)
                break;
            for (auto && el : data) {
                ofs << int(el) << " ";
            }
            ofs << "\n";

            uint8_t label;
            bin_read(ifs_o, label);
            if (!ifs_o)
                break;

            for (int i = 0; i < 10; ++i) {
                if(i == label)
                    ofs << int(1) << " ";
                else
                    ofs << int(0) << " ";
            }
            ofs << "\n";

            if (!ifs_i || !ifs_o)
                break;
        }        
    }
    */
    
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
    

    LearningSet trainingSet = LearningSet::readFannSet(R"(c:\users\luca\desktop\minst.fann_train)");
    LearningSet testSet = LearningSet::readFannSet(R"(c:\users\luca\desktop\minst.fann_test)");
    //trainingSet.writeFannSet(R"(c:\users\luca\desktop\minst.fann_train)");
    //testSet.writeFannSet(R"(c:\users\luca\desktop\minst.fann_test)");
    
    std::vector<int> layers={28 * 28,300,100,10};
    std::vector<int> functs = { -1,3,3,1 };
    
    
    /*
    {
        std::string trai = R"(
    3 2 1
    7 4
    3
    2 9
    5
    33 27
    6
    )";
        std::ofstream tt("tt.fann");
        tt << trai;
    }

    LearningSet trainingSet = LearningSet::readFannSet("tt.fann");
    LearningSet testSet = LearningSet::readFannSet("tt.fann");
    std::vector<int> layers = { 2, 1 };
    std::vector<int> functs = { 3, 0 };
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
    /*
    LearningSet trainingSet = LearningSet::readBinarySet("adi_train.set");
    int n = 0;
    trainingSet = trainingSet.shuffle().filter_set_by([&](auto&&, auto &&, auto &&) {return n++ < 400 * 4; });
    //trainingSet = trainingSet.shuffle();
    LearningSet testSet = LearningSet::readBinarySet("adi_test.set");
    //std::vector<LearningSet> batches;
    //n = 0;
    //trainingSet = trainingSet.filter_set_by([&](auto, auto, auto) {++n; return n < 400*4; });
    
    ////for (int i = 0; i < trainingSet.getNumOfInstances(); i += 300 * 4) {
    ////    n = 0;
    ////    batches.push_back(trainingSet.filter_set_by([&] (auto,auto,auto){++n; return n > i && n < i + 1200; }));
    ////}
    ////trainingSet = LearningSet();
    
    //LearningSet trainingSet(R"(C:\Users\Luca\Desktop\adidas_project\trunk\vision_code\feature_extraction.build\train.set)");
    //LearningSet testSet(R"(C:\Users\Luca\Desktop\adidas_project\trunk\vision_code\feature_extraction.build\train.set)");
    std::vector<int> layers = { 200 * 200 * 3, 700,500,300,100,4 };
    std::vector<int> functs={-1,3,3,3,3,1};
    */
/*
    LearningSet trainingSet = LearningSet::readFannSet(std::string(base + "tom_train.set").c_str());
    LearningSet testSet = LearningSet::readFannSet(std::string(base + "tom_test.set").c_str());
    std::vector<int> layers = { 1500, 800, 500,2 };
    std::vector<int> functs = { 3,3,3,1 };


    //layer sizes
    //activation functions (1=sigm,2=tanh,3=relu)
    */
    
    //declare the network with the number of layers
    //FeedForwardNN mynet(3, layers, functs);
    FeedForwardNN mynet(layers.size(),&layers[0],&functs[0]);
    mynet.initWeightsUniform(-0.01, 0.01);
    //mynet.initWeightsGaussian(1.0, 0.01);
    //mynet.initWeightsWidrowNguyen(testSet);
   mynet.initWeightsBengio(0.1);
    
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
   
    //trainer.selectTrainingSet(trainingSet);
    // mynet.initWidrowNguyen(trainingSet);

    TrainingParameters params;
    params.training_location = TRAIN_GPU;
    params.training_algorithm = ALG_BATCH;
    params.desired_error = 0.0;
    params.max_epochs = 100000;
    params.epochs_between_reports = 20;
    params.learningRate = 1.;
    //params.learningRate = 0.01;
    params.momentum = 0.0;
    params.shuff = SHUFFLE_ON;
    params.errorFunc = ERROR_LINEAR;

    trainer.train(params);
    
    mynet.saveToTxt("mynetmushrooms.net");
    
    //mseT.saveToTxt("../mseTmushrooms.net");
    //mseTT.saveToTxt("../mseTTmushrooms.net");
    cl.saveToTxt("clmushrooms.net");
    std::cout << "saved" << "\n";

    //test_classification(cl, testSet);
    //test_all(cl, testSet);

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
    std::cout << out[0] << "\n";
*/
    return 0;
}

int main() {
    //showimg(std::vector<int>{10, 40, 100, 1000}, 2, 2);
    //showimg(std::vector<int>{10, 40, 100, 1000}, 2, 2);
   

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
    /*
    LearningSet trainingSet;
    LearningSet testSet;
    
    {
        LearningSet no = LearningSet::readInOutSet(R"(C:\Users\Luca\Desktop\textureono\no.set)", R"(C:\Users\Luca\Desktop\textureono\zero.set)");
        LearningSet tex = LearningSet::readInOutSet(R"(C:\Users\Luca\Desktop\textureono\tex.set)", R"(C:\Users\Luca\Desktop\textureono\one.set)");


        LearningSet allSet = no;
        allSet.insert(tex);
        allSet = allSet.shuffle();

        int n = 0;
        trainingSet = allSet.filter_set_by([&](auto &&, auto &&, auto &&) {
            ++n;
            return n <= 1000;
        });
        n = 0;
        testSet = allSet.filter_set_by([&](auto &&, auto &&, auto &&) {
            ++n;
            return n > 1000 && n < 1100;
        });
    }
    //trainingSet.writeFannSet(R"(c:\users\luca\desktop\minst.fann_train)");
    //testSet.writeFannSet(R"(c:\users\luca\desktop\minst.fann_test)");

    std::vector<int> layers = { 200 * 200 * 3,300,100, 30 };

    std::vector<int> functs = { -1,3,3,3 };
    for (int i = 0; i <5; ++i) {
        // layers.push_back((10-i)*50);
        //layers.push_back(40);
        //functs.push_back(3);

    }
    layers.push_back(2);
    functs.push_back(1);
    */
    /*
    auto trainingSet = LearningSet::readFannSet("parity13.train");
    auto testSet = LearningSet::readFannSet("parity13.test");
    std::vector<int> layers = { 13, 100,30, 1 };
    std::vector<int> functs = { -1,3,3,1 };
    auto * ins = trainingSet.getInputs();
    int num = trainingSet.getNumOfInstances() * trainingSet.getNumOfInputsPerInstance();
    for (int i = 0; i < num; ++i) {
        ins[i] -= 0.5;
    }
    */

    /*
    std::ofstream ts("aaa.txt");
    ts << "8 10 -16 4\n";
    ts << "7\n";
    ts << "4 5 -8 2\n";
    ts << "3.5\n";
    ts.close();
    auto trainingSet = LearningSet::readSimplifiedSet("aaa.txt");
    std::vector<int> layers = { 4, 2, 1};
    std::vector<int> functs = { -1,0,0};
    */
    
    
    LearningSet trainingSet = LearningSet::readFannSet(R"(c:\users\luca\desktop\minst.fann_train)");
    LearningSet testSet = LearningSet::readFannSet(R"(c:\users\luca\desktop\minst.fann_test)");
    //trainingSet.writeFannSet(R"(c:\users\luca\desktop\minst.fann_train)");
    //testSet.writeFannSet(R"(c:\users\luca\desktop\minst.fann_test)");

    std::vector<int> layers = { 28 * 28,300,100};

    std::vector<int> functs = { -1,3,3};
    for (int i = 0; i <10; ++i) {
// layers.push_back((10-i)*50);
        //layers.push_back(50);
       //functs.push_back(3);

    }
    layers.push_back(10);
    functs.push_back(1);
    
    

    /*
    {
    std::string trai = R"(
    3 2 1
    7 4
    3
    2 9
    5
    33 27
    6
    )";
    std::ofstream tt("tt.fann");
    tt << trai;
    }

    LearningSet trainingSet = LearningSet::readFannSet("tt.fann");
    LearningSet testSet = LearningSet::readFannSet("tt.fann");
    std::vector<int> layers = { 2, 1 };
    std::vector<int> functs = { 3, 0 };
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
    /*
    LearningSet trainingSet = LearningSet::readBinarySet("adi_train.set");
    int n = 0;
    trainingSet = trainingSet.shuffle().filter_set_by([&](auto&&, auto &&, auto &&) {return n++ < 400 * 4; });
    //trainingSet = trainingSet.shuffle();
    LearningSet testSet = LearningSet::readBinarySet("adi_test.set");
    //std::vector<LearningSet> batches;
    //n = 0;
    //trainingSet = trainingSet.filter_set_by([&](auto, auto, auto) {++n; return n < 400*4; });

    ////for (int i = 0; i < trainingSet.getNumOfInstances(); i += 300 * 4) {
    ////    n = 0;
    ////    batches.push_back(trainingSet.filter_set_by([&] (auto,auto,auto){++n; return n > i && n < i + 1200; }));
    ////}
    ////trainingSet = LearningSet();

    //LearningSet trainingSet(R"(C:\Users\Luca\Desktop\adidas_project\trunk\vision_code\feature_extraction.build\train.set)");
    //LearningSet testSet(R"(C:\Users\Luca\Desktop\adidas_project\trunk\vision_code\feature_extraction.build\train.set)");
    std::vector<int> layers = { 200 * 200 * 3, 700,500,300,100,4 };
    std::vector<int> functs={-1,3,3,3,3,1};
    */
    /*
    LearningSet trainingSet = LearningSet::readFannSet(std::string(base + "tom_train.set").c_str());
    LearningSet testSet = LearningSet::readFannSet(std::string(base + "tom_test.set").c_str());
    std::vector<int> layers = { 1500, 800, 500,2 };
    std::vector<int> functs = { 3,3,3,1 };


    //layer sizes
    //activation functions (1=sigm,2=tanh,3=relu)
    */


    //declare the network with the number of layers
    //FeedForwardNN mynet(3, layers, functs);
    FeedForwardNN mynet(layers.size(), &layers[0], &functs[0]);
    mynet.initWeightsUniform(-0.0010, 0.0010);
    //mynet.initWeightsGaussian(0.00001, 1.0);
    //mynet.initWeightsWidrowNguyen(testSet);
    //mynet.initWeightsDonati(0.00001);
    //mynet.initWeightsBengio(0.00001);
    /*
    auto * w = mynet.getWeights();
    auto tot = mynet.getNumOfWeights();
    w[0] = 1;
    w[1] = -0.6;
    w[2] = 0;
    w[3] = 0;

    w[4] = 0;

    w[5] = 0;
    w[6] = 0;
    w[7] = 0.1;
    w[8] = 0.9;

    w[9] = 0;


    w[10] = 0.8;
    w[11] = 0.7;
    
    w[12] = 0;
    */
    

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

    //trainer.selectTrainingSet(trainingSet);
    // mynet.initWidrowNguyen(trainingSet);

    TrainingParameters params;
    params.training_location = TRAIN_GPU;
    params.training_algorithm = ALG_BATCH;
    params.desired_error = 0.0;
    params.max_epochs = 500000;
    params.epochs_between_reports = 20;
    //params.learningRate = 0.31;
    params.learningRate = 0.00001;
    params.momentum = 0.9;// 99;
    params.shuff = SHUFFLE_ON;
    params.errorFunc = ERROR_LINEAR;

 //   while (true) {
    trainer.train(params);
//        mynet.addWeightsDonati(-6, -2);
 //   }

    //    trainer.train(params);

    mynet.saveToTxt("mynetmushrooms.net");

    //mseT.saveToTxt("../mseTmushrooms.net");
    //mseTT.saveToTxt("../mseTTmushrooms.net");
    cl.saveToTxt("clmushrooms.net");
    std::cout << "saved" << "\n";

    test_classification(cl, testSet);
    //test_all(cl, testSet);
    //test_all(mynet, trainingSet);

    getchar();

}




