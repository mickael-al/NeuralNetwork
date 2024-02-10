#include "LinearModel.hpp"
#include <ctime>
#include <iostream>
#include <algorithm>

LinearModel::LinearModel() 
{
    std::srand(static_cast<unsigned int>(std::time(0)));
    m_weight = new double[3];
    std::generate(m_weight, m_weight + 3, []() { return ((static_cast<float>(rand()) / RAND_MAX) * 2.0f) - 1.0f; });
}

void LinearModel::training(float learning_rate, Vec2* training_data, int size, std::vector<double> point, std::vector<float>* error) 
{
    std::srand(static_cast<unsigned int>(std::time(0)));
    float errorCount = 0.0f;
    int modulo = std::max(size,50);
    float mf = modulo;
    for (int i = 0; i < 1000000; i++) 
    {
        int k = i % size;
        double yk = point[k];
        Vec3 xk(1.0, training_data[k].x, training_data[k].y);
        Vec3 weight(m_weight[0], m_weight[1], m_weight[2]);
        double dot = weight.x * xk.x + weight.y * xk.y + weight.z * xk.z;
        double gXk = dot >= 0 ? 1.0 : -1.0;
        m_weight[0] += learning_rate * (yk - gXk) * xk.x;
        m_weight[1] += learning_rate * (yk - gXk) * xk.y;
        m_weight[2] += learning_rate * (yk - gXk) * xk.z;
        double p = predict(&training_data[k]);
        errorCount += (p >= 0 && yk >= 0) || (p < 0 && yk < 0) ? 0 : 100;
        if (i % modulo == (modulo-1))
        {            
            error->push_back(errorCount/ mf);
            if ((errorCount / mf) <= 1.0f)
            {
                break;
            }
            errorCount = 0.0f;
        }
    }
}

double LinearModel::predict(Vec2* point) 
{
    return m_weight[0] + m_weight[1] * point->x + m_weight[2] * point->y;
}

LinearModel::~LinearModel() 
{
    delete[] m_weight;
}
