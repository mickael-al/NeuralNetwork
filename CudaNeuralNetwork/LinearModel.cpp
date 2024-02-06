#include "LinearModel.hpp"
#include <ctime>
#include <iostream>

LinearModel::LinearModel()
{
	std::srand(static_cast<unsigned int>(std::time(0)));	
	m_weight = new double[3];
	for (int i = 0; i < 3; i++)
	{
		m_weight[i] = ((static_cast<float>(rand()) / RAND_MAX) * 2.0f) - 1.0f;
	}
}

void LinearModel::training(float learning_rate, Vec2* training_data,int size, std::vector<double> point)
{
	std::srand(static_cast<unsigned int>(std::time(0)));
	for (int i = 0; i < 1000000; i++)
	{
		int k = (static_cast<int>(rand()) / RAND_MAX) % size;		
		double yk = point[k];
		Vec3 xk(1.0, training_data[k].x, training_data[k].y);
		Vec3 weight(m_weight[0], m_weight[1], m_weight[2]);
		double dot = weight.x * xk.x + weight.y * xk.y + weight.z * xk.z;
		double gXk = dot >= 0 ? 1.0 : -1.0;
		m_weight[0] += learning_rate * (yk - gXk) * xk.x;
		m_weight[1] += learning_rate * (yk - gXk) * xk.y;
		m_weight[2] += learning_rate * (yk - gXk) * xk.z;
	}
}

double LinearModel::predict(Vec2* point)
{
	return m_weight[0] + m_weight[1] * point->x / 100.0 + m_weight[2] * point->y / 100.0;
}

LinearModel::~LinearModel()
{
	delete m_weight;
}