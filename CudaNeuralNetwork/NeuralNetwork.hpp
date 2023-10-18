#ifndef __NEURAL_NETWORK__
#define __NEURAL_NETWORK__

#include <iostream>
#include <vector>

class NeuralNetwork 
{
public:
    void loadModel(const std::string& modelPath);
    void setInputData(const std::vector<double>& inputData);
    void saveModel(const std::string& modelPath);
    void Propagate();
    void BackPropagate();
    std::vector<double> predict();    
};

#endif //!__NEURAL_NETWORK__