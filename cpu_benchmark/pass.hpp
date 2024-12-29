#ifndef PASS_H_
#define PASS_H_

#define NUM_THREADS 4

#include <array>
#include <vector>
#include <Eigen/Dense>


class NodeArr {
    public:
    
    //Setting activations to be a little larger to zero pad all operations
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> activations; 
    //Note thate we are making the vector of weights 9 so that we can get blocks of activations
    std::vector<std::vector<Eigen::Vector<float, 9>>> NodeWeights;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> bias;
    
    
    NodeArr(int ARR_DIM_);
    void setWeight(int x, int y, int k, float weight);
    void setBias(int x, int y, float weight);
    void setActivation(int x, int y, float activation);
    void update();
    void update_par1();
    void update_par2();
    float getActivation(int x, int y);

    int ARR_DIM;

};

#endif  // PASS_H_
