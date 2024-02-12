#include "Rbf.hpp"
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

Rbf::Rbf(size_t nb_poids, double gam, std::vector<std::vector<double>> x, std::vector<double> y) : poids(std::vector<double>(nb_poids, 0.0)), exemple_x(std::move(x)), exemple_y(std::move(y)), gammma(gam) 
{
    size_t n = exemple_x.size();
    Eigen::MatrixXd matrix(n, n);

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            double diff_x1 = exemple_x[i][0] - exemple_x[j][0];
            double diff_x2 = exemple_x[i][1] - exemple_x[j][1];
            double mag_x = diff_x1 * diff_x1 + diff_x2 * diff_x2;
            matrix(i, j) = std::exp(-gammma * mag_x);
        }
    }

    Eigen::MatrixXd matrix_inv = matrix.inverse();

    std::vector<std::vector<double>> matrix_inv_vec(n, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            matrix_inv_vec[i][j] = matrix_inv(i, j);
        }
    }

    std::vector<double> new_poids;

    for (const auto& row : matrix_inv_vec)
    {
        double dot_product = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot_product += row[i] * exemple_y[i];
        }
        new_poids.push_back(dot_product);
    }

    poids = new_poids;
}

double Rbf::predict(double x1, double x2) const 
{
    double output = 0.0;
    for (size_t j = 0; j < exemple_x.size(); ++j) 
    {
        double diff_x1 = x1 - exemple_x[j][0];
        double diff_x2 = x2 - exemple_x[j][1];
        double mag_x = diff_x1 * diff_x1 + diff_x2 * diff_x2;
        output += poids[j] * std::exp(-gammma * mag_x);
    }
    return output;
}
