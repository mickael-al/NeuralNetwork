//Peer Programming: Guo, Albarello
#ifndef LINEAR_MODEL_HPP
#define LINEAR_MODEL_HPP

#include <vector>
#include "Vec.hpp"

class LinearModel 
{
public:
    LinearModel();
    ~LinearModel();
    void training(double learning_rate, Vec2* training_data, int size, std::vector<double> point, std::vector<float>* error);
    double predict(Vec2* point);

private:
    double* m_weight;
};

#endif // LINEAR_MODEL_HPP
