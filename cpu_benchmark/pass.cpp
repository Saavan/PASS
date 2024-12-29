
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include "pass.hpp"
#include <ctime>




float sigmoid(float x) {
    return 1/(1 + exp(-1 * x));
} 

NodeArr::NodeArr(int ARR_DIM_) : ARR_DIM(ARR_DIM_), NodeWeights(ARR_DIM_), activations(ARR_DIM_ + 2, ARR_DIM_ + 2), bias(ARR_DIM_, ARR_DIM_) {
    activations.setZero();
    bias.setZero();
    for(int i = 0; i < ARR_DIM; i++) {
	NodeWeights[i] = std::vector<Eigen::Vector<float, 9>>(ARR_DIM);
        for(int j = 0; j < ARR_DIM; j++) {
            NodeWeights[i][j].setZero();
        }
    }

}

void NodeArr::setWeight(int x, int y, int k, float weight) {
    NodeWeights[x][y][k] = weight;
}

void NodeArr::setBias(int x, int y, float weight) {
    bias(x, y) = weight;
}

float NodeArr::getActivation(int x, int y) {
    return activations(x + 1, y + 1);
}

void NodeArr::setActivation(int x, int y, float activation) {
    activations(x + 1, y + 1) = activation;
}

void NodeArr::update() {
    Eigen::MatrixXf randFloats = Eigen::MatrixXf::Random(ARR_DIM, ARR_DIM); 
    for(int i = 1; i < ARR_DIM + 1; i++) {
        for(int j = 1; j < ARR_DIM + 1; j++) {
            auto neighbors = activations(Eigen::seq(i-1, i+1), Eigen::seq(j-1, j+1)).reshaped();
            float act_val = neighbors.dot(NodeWeights[i-1][j-1]) + bias(i-1, j-1);
            activations(i, j) = sigmoid(act_val) > randFloats(i-1, j-1);
        }
    }
}

void NodeArr::update_par1() {
    Eigen::MatrixXf randFloats = Eigen::MatrixXf::Random(ARR_DIM, ARR_DIM); 

    #pragma omp parallel for private(activations)
    for(int k = 0; k < 2; k++) {
        for(int l = 0; l < 2; l++) {
            //Block parallel loops
            for(int i = 1; i < ARR_DIM + 1; i+=2) {
                for(int j = 1; j < ARR_DIM + 1; j+=2) {
		    std::cout << k + i - 1 << ", " << k + i + 1 << ", " << l + j - 1 << ", " << l + j + 1 << std::endl;
                    auto neighbors = activations(Eigen::seq(k + i-1, k + i+1), Eigen::seq(l + j-1, l + j+1)).reshaped();
                    float act_val = neighbors.dot(NodeWeights[k + i-1][l + j-1]) + bias(k + i-1, l + j-1);
                    activations(k+i, l+j) = sigmoid(act_val) > randFloats(k+i-1, l+j-1);
                }
            }
            
        }
    }
}

void NodeArr::update_par2() {
    Eigen::MatrixXf randFloats = Eigen::MatrixXf::Random(ARR_DIM, ARR_DIM); 

    for(int k = 0; k < 2; k++) {
        for(int l = 0; l < 2; l++) {
            //Block parallel loops
            #pragma omp parallel for collapse(2) private(activations)
            for(int i = 1; i < ARR_DIM + 1; i+=2) {
                for(int j = 1; j < ARR_DIM + 1; j+=2) {
                    auto neighbors = activations(Eigen::seq(k + i-1, k + i+1), Eigen::seq(l + j-1, l + j+1)).reshaped();
                    float act_val = neighbors.dot(NodeWeights[k + i-1][l + j-1]) + bias(k + i-1, l + j-1);
                    activations(k+i, l+j) = sigmoid(act_val) > randFloats(i-1, j-1);
                }
            }
            
        }
    }
}

int main (int argc, char *argv[]) {
    int ARR_DIM = 16;
    int NUM_RUNS = 10000;
    if(argc > 1) {
	ARR_DIM = std::atoi(argv[1]);	
    } 
    if(argc > 2) {
	NUM_RUNS = std::atoi(argv[2]);
    }
    NodeArr pass(ARR_DIM);
    NodeArr pass_par(ARR_DIM);
    omp_set_num_threads(NUM_THREADS);
    for(int i = 0; i < ARR_DIM; i++) {
        for(int j = 0; j < ARR_DIM; j++) {
            for(int k = 0; k < 8; k++) {
                pass.setWeight(i, j, k, 1);
                pass_par.setWeight(i, j, k, 1);
                pass.setBias(i, j, 10);
                pass_par.setBias(i, j, 10);
            }
        }
    }


    for(int i = 0; i < ARR_DIM; i++) {
        for(int j = 0; j < ARR_DIM; j++) {
            pass.setActivation(i, j, i*ARR_DIM + j);
            pass_par.setActivation(i, j, i*ARR_DIM + j);
        }
    }


    std::cout << "Timing regular update" << std::endl;
    double duration;
    std::clock_t start = std::clock();
    for(int i = 0; i < NUM_RUNS; i++) {
        pass.update();
    }
    std::clock_t end = std::clock();
    duration = ( end - start ) / (double) CLOCKS_PER_SEC;
    std::cout << "Number of Runs:" << NUM_RUNS << " Total Time:" << duration << " Time per run:" << (duration/((float)NUM_RUNS)) << std::endl; 


    std::cout << "Timing parallel update 1" << std::endl;
    start = std::clock();
    for(int i = 0; i < NUM_RUNS; i++) {
        pass.update_par1();
    }
    end = std::clock();
    duration = ( end - start ) / (double) CLOCKS_PER_SEC;
    std::cout << "Number of Runs:" << NUM_RUNS << " Total Time:" << duration << " Time per run:" << (duration/((float)NUM_RUNS)) << std::endl; 

    std::cout << "Timing parallel update 1" << std::endl;
    start = std::clock();
    for(int i = 0; i < NUM_RUNS; i++) {
        pass.update_par2();
    }
    end = std::clock();
    duration = ( end - start ) / (double) CLOCKS_PER_SEC;
    std::cout << "Number of Runs:" << NUM_RUNS << " Total Time:" << duration << " Time per run:" << (duration/((float)NUM_RUNS)) << std::endl; 


}
