#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  // TODO get laser working
  use_laser_ = false;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1; //30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI / 4; //30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  is_initialized_ = false;

  n_x_ = 5;
  n_aug_ = 7;

  lambda_ = 3 - n_aug_;

  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(0.5 / (lambda_ + n_aug_));
  weights_(0) = 1. * lambda_ / (lambda_ + n_aug_);

  Q_ = MatrixXd(2, 2);
  Q_ << std_a_ * std_a_, 0,
        0, std_yawdd_ * std_yawdd_;

  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
              0, std_radphi_ * std_radphi_, 0,
              0, 0, std_radrd_ * std_radrd_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  VectorXd raw_meas = meas_package.raw_measurements_;

  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_[0] = raw_meas[0];
      x_[1] = raw_meas[1];
      x_[2] = 0;
      x_[3] = 0;
      x_[4] = 0;
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      double x_proj = cos(raw_meas[1]);
      double y_proj = sin(raw_meas[1]);
      x_[0] = raw_meas[0] * x_proj;
      x_[1] = raw_meas[0] * y_proj;
      x_[2] = 0;
      x_[3] = 0;
      x_[4] = 0;
    }

    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1000, 0, 0,
          0, 0, 0, 1000, 0,
          0, 0, 0, 0, 1000;

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  long long curr_time_us = meas_package.timestamp_;
  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    double delta_t = (curr_time_us - time_us_) / 1000000.0;
    Prediction(delta_t);
    UpdateLidar(meas_package);
    time_us_ = curr_time_us;
  } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    double delta_t = (curr_time_us - time_us_) / 1000000.0;
    Prediction(delta_t);
    UpdateRadar(meas_package);
    time_us_ = curr_time_us;
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

  x_aug.head(x_.size()) = x_;
  P_aug.topLeftCorner(P_.rows(), P_.cols()) = P_;
  P_aug.bottomRightCorner(Q_.rows(), Q_.cols()) = Q_;

  MatrixXd A_aug = P_aug.llt().matrixL();

  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i + 1)          = x_aug + sqrt(lambda_ + n_aug_) * A_aug.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A_aug.col(i);
  }

  Xsig_pred_ = MatrixXd(n_x_, n_aug_ * 2 + 1);

  for (int i = 0; i < Xsig_aug.cols(); ++i) {
    VectorXd x_aug_col = Xsig_aug.col(i);
    const double px        = x_aug_col(0);
    const double py        = x_aug_col(1);
    const double v         = x_aug_col(2);
    const double psi       = x_aug_col(3);
    const double psi_dot   = x_aug_col(4);
    const double nu_a      = x_aug_col(5);
    const double nu_psi_dd = x_aug_col(6);

    VectorXd x_pred = VectorXd(n_x_);
    if (fabs(psi_dot) < 0.001) {
      x_pred(0) = v * cos(psi) * delta_t;
      x_pred(1) = v * sin(psi) * delta_t;
      x_pred(2) = 0;
      x_pred(3) = 0; // psi_dot * dt is always zero here
      x_pred(4) = 0;
    } else {
      const double c = (v / psi_dot);
      const double psi_dot_dt = psi_dot * delta_t;
      const double psi_plus_psi_dot_dt = psi + psi_dot_dt;
      x_pred(0) = c * (+sin(psi_plus_psi_dot_dt) - sin(psi));
      x_pred(1) = c * (-cos(psi_plus_psi_dot_dt) + cos(psi));
      x_pred(2) = 0;
      x_pred(3) = psi_dot_dt;
      x_pred(4) = 0;
    }

    VectorXd x_k = x_aug.head(n_x_);
    VectorXd x_noise = VectorXd(n_x_);
    const double half_dt2 = 0.5 * delta_t * delta_t;
    x_noise(0) = half_dt2 * cos(psi) + nu_a;
    x_noise(1) = half_dt2 * sin(psi) + nu_a;
    x_noise(2) = delta_t * nu_a;
    x_noise(3) = half_dt2 * nu_psi_dd;
    x_noise(4) = delta_t * nu_psi_dd;
    Xsig_pred_.col(i) = x_k + x_pred + x_noise;
  }

  VectorXd x_pred = VectorXd::Zero(n_x_);
  MatrixXd P_pred = MatrixXd::Zero(n_x_, n_x_);
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    x_pred += weights_(i) * Xsig_pred_.col(i);
  }

  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    MatrixXd x_diff = Xsig_pred_.col(i) - x_pred;
    x_diff(3) = tools.NormalizeAngle(x_diff(3));
    MatrixXd x_diff_t = x_diff.transpose();
    P_pred += weights_(i) * x_diff * x_diff_t;
  }

  x_ = x_pred;
  P_ = P_pred;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  // radar can measure r, phi, and r_dot
  const int n_z = 3;

  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    VectorXd xsig_pred = Xsig_pred_.col(i);
    const double px = xsig_pred(0);
    const double py = xsig_pred(1);
    const double v = xsig_pred(2);
    const double psi = xsig_pred(3);
    const double p_mag = sqrt(px * px + py * py);

    VectorXd zsig = VectorXd(n_z);
    zsig(0) = p_mag;
    zsig(1) = atan2(py, px);
    zsig(2) = (px * cos(psi) * v + py * sin(psi) * v) / p_mag;
    Zsig.col(i) = zsig;
  }

  VectorXd z_pred = VectorXd::Zero(n_z);
  for (int i = 0; i < Zsig.cols(); ++i) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  MatrixXd S = MatrixXd::Zero(n_z, n_z);
  for (int i = 0; i < Zsig.cols(); ++i) {
    MatrixXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = tools.NormalizeAngle(z_diff(1));
    S += weights_(i) * z_diff * z_diff.transpose();
  }
  S += R_radar_;

  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = tools.NormalizeAngle(z_diff(1));
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = tools.NormalizeAngle(x_diff(3));
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = Tc * S.inverse();

  VectorXd raw_meas = meas_package.raw_measurements_;
  VectorXd z = VectorXd(3);
  z << raw_meas[0],
       raw_meas[1],
       raw_meas[2];
  VectorXd z_diff = z - z_pred;
  z_diff(1) = tools.NormalizeAngle(z_diff(1));

  x_ = x_ + K * z_diff;
  x_(3) = tools.NormalizeAngle(x_(3));
  P_ = P_ - K * S * K.transpose();

  std::cout << z << "\n\n" << x_ << "\n\n" << P_ << "\n\n\n\n\n";
}
