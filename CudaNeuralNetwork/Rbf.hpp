//Peer Programming: Guo, Albarello
#ifndef __RBF__
#define __RBF__

#include <vector>

class Rbf 
{
public:
    Rbf(size_t nb_poids, double gam, std::vector<std::vector<double>> x, std::vector<double> y);

    double predict(double x1, double x2) const;
private:
    std::vector<double> poids;
    std::vector<std::vector<double>> exemple_x;
    std::vector<double> exemple_y;
    double gammma;
};
#endif//!__RBF__