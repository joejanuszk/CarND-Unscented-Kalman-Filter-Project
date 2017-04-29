#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  int estimations_size = estimations.size();

  if (estimations_size != ground_truth.size()) {
    std::cout << "Estimations and ground truth must be equal size\n";
    return rmse;
  }

  if (estimations_size == 0) {
    std::cout << "Cannot calculate RMSE of 0 estimations\n";
    return rmse;
  }

  for (int i = 0; i < estimations_size; ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  rmse = rmse / estimations_size;
  rmse = rmse.array().sqrt();
  return rmse;
}

double Tools::NormalizeAngle(double angle) {
  while (angle > +M_PI) angle -= 2. * M_PI;
  while (angle < -M_PI) angle += 2. * M_PI;
  return angle;
}
