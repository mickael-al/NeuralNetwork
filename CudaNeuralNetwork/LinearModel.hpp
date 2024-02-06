#ifndef __LINEAR_MODEL__
#define __LINEAR_MODEL__

#include <vector>
#include "Vec.hpp"

class LinearModel
{
public:
	LinearModel();
	~LinearModel();
	void training(float learning_rate, Vec2 * training_data,int size,std::vector<double> point);
	double predict(Vec2*point);
private:
	double * m_weight;	
};

#endif //!__LINEAR_MODEL__

