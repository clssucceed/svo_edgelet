// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <svo/feature.h>
#include <svo/frame.h>
#include <svo/math_lib.h>
#include <svo/point.h>
#include <svo/pose_optimizer.h>
#include <svo/robust_cost.h>
#include <stdexcept>

namespace svo {
namespace pose_optimizer {
/*
void optimizeGaussNewton(
    const double reproj_thresh,
    const size_t n_iter,
    const bool verbose,
    FramePtr& frame,
    double& estimated_scale,
    double& error_init,
    double& error_final,
    size_t& num_obs)
{
  // init
  double chi2(0.0);
  vector<double> chi2_vec_init, chi2_vec_final;
  svo::robust_cost::TukeyWeightFunction weight_function;
  SE3 T_old(frame->T_f_w_);
  Matrix6d A;
  Vector6d b;

  // compute the scale of the error for robust estimation
  std::vector<float> errors; errors.reserve(frame->fts_.size());
  for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
  {
    if((*it)->point == NULL)
      continue;
    Vector2d e = project2d((*it)->f)
               - project2d(frame->T_f_w_ * (*it)->point->pos_);
    e *= 1.0 / (1<<(*it)->level);
    errors.push_back(e.norm());
  }
  if(errors.empty())
    return;
  svo::robust_cost::MADScaleEstimator scale_estimator;
  estimated_scale = scale_estimator.compute(errors);

  num_obs = errors.size();
  chi2_vec_init.reserve(num_obs);
  chi2_vec_final.reserve(num_obs);
  double scale = estimated_scale;
  for(size_t iter=0; iter<n_iter; iter++)
  {
    // overwrite scale
    if(iter == 5)
      scale = 0.85/frame->cam_->errorMultiplier2();

    b.setZero();
    A.setZero();
    double new_chi2(0.0);

    // compute residual
    for(auto it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
    {
      if((*it)->point == NULL)
        continue;
      Matrix26d J;
      Vector3d xyz_f(frame->T_f_w_ * (*it)->point->pos_);
      Frame::jacobian_xyz2uv(xyz_f, J);
      Vector2d e = project2d((*it)->f) - project2d(xyz_f);
      double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
      e *= sqrt_inv_cov;
      if(iter == 0)
        chi2_vec_init.push_back(e.squaredNorm()); // just for debug
      J *= sqrt_inv_cov;
      double weight = weight_function.value(e.norm()/scale);
      A.noalias() += J.transpose()*J*weight;
      b.noalias() -= J.transpose()*e*weight;
      new_chi2 += e.squaredNorm()*weight;
    }

    // solve linear system
    const Vector6d dT(A.ldlt().solve(b));

    // check if error increased
    if((iter > 0 && new_chi2 > chi2) || (bool) std::isnan((double)dT[0]))
    {
      if(verbose)
        std::cout << "it " << iter
                  << "\t FAILURE \t new_chi2 = " << new_chi2 << std::endl;
      frame->T_f_w_ = T_old; // roll-back
      break;
    }

    // update the model
    SE3 T_new = SE3::exp(dT)*frame->T_f_w_;
    T_old = frame->T_f_w_;
    frame->T_f_w_ = T_new;
    chi2 = new_chi2;
    if(verbose)
      std::cout << "it " << iter
                << "\t Success \t new_chi2 = " << new_chi2
                << "\t norm(dT) = " << norm_max(dT) << std::endl;

    // stop when converged
    if(norm_max(dT) <= EPS)
      break;
  }

  // Set covariance as inverse information matrix. Optimistic estimator!
  const double pixel_variance=1.0;
  frame->Cov_ =
pixel_variance*(A*std::pow(frame->cam_->errorMultiplier2(),2)).inverse();

  // Remove Measurements with too large reprojection error
  double reproj_thresh_scaled = reproj_thresh / frame->cam_->errorMultiplier2();
  size_t n_deleted_refs = 0;
  for(Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
  {
    if((*it)->point == NULL)
      continue;
    Vector2d e = project2d((*it)->f) - project2d(frame->T_f_w_ *
(*it)->point->pos_);
    double sqrt_inv_cov = 1.0 / (1<<(*it)->level);
    e *= sqrt_inv_cov;
    chi2_vec_final.push_back(e.squaredNorm());
    if(e.norm() > reproj_thresh_scaled)
    {
      // we don't need to delete a reference in the point since it was not
created yet
      (*it)->point = NULL;
      ++n_deleted_refs;
    }
  }

  error_init=0.0;
  error_final=0.0;
  if(!chi2_vec_init.empty())
    error_init = sqrt(getMedian(chi2_vec_init))*frame->cam_->errorMultiplier2();
  if(!chi2_vec_final.empty())
    error_final =
sqrt(getMedian(chi2_vec_final))*frame->cam_->errorMultiplier2();

  estimated_scale *= frame->cam_->errorMultiplier2();
  if(verbose)
    std::cout << "n deleted obs = " << n_deleted_refs
              << "\t scale = " << estimated_scale
              << "\t error init = " << error_init
              << "\t error end = " << error_final << std::endl;
  num_obs -= n_deleted_refs;
}
*/

#define EDGE
#ifdef EDGE
// reproj_thresh: Reprojection threshold after pose optimization
// n_iter: Number of iterations in local bundle adjustment
// verbose: false
// frame: current frame
// estimated_scale: 优化前的所有重投影误差的median
// error_init: 优化前的重投影误差的median
// error_final: 优化后的重投影误差的median
// num_obs: 优化后重投影误差小于阈值的特征数目
// 为什么优化时计算重投影误差没有乘以focal_length?
// 因为所有重投影误差不乘以focal_length和乘以focal_length是等价的
void optimizeGaussNewton(const double reproj_thresh, const size_t n_iter,
                         const bool verbose, FramePtr& frame,
                         double& estimated_scale, double& error_init,
                         double& error_final, size_t& num_obs) {
  // init
  double chi2(0.0);
  // 保存每一个feature优化前和优化后的所有的res^2
  vector<double> chi2_vec_init, chi2_vec_final;
  // 设置robust kernel,减小outlier对于优化结果的影响
  svo::robust_cost::TukeyWeightFunction weight_function;
  SE3 T_old(frame->T_f_w_);  // pose from 2d vs 2d photometric alignment as init
  Matrix6d A;
  Vector6d b;

  // compute the scale of the error for robust estimation
  std::vector<float> errors;
  errors.reserve(frame->fts_.size());
  // 遍历所有reprojectMap(3d vs 2d alignment)中跟踪成功的3d
  // feature计算所有特征的重投影误差
  for (auto it = frame->fts_.begin(); it != frame->fts_.end(); ++it) {
    // 跳过没有成功三角化的特征
    if ((*it)->point == NULL) continue;
    // 当前feature在z=1平面上的重投影误差
    Vector2d e =
        project2d((*it)->f) - project2d(frame->T_f_w_ * (*it)->point->pos_);

    // scale到search_level_(非常接近于特征提取时的scale)
    e *= 1.0 / (1 << (*it)->level);  // weight  the error  according which level
                                     // the feature locate

    if ((*it)->type == Feature::EDGELET) {
      // e在grad上的分量
      errors.push_back(std::fabs((*it)->grad.transpose() * e));
    } else {
      errors.push_back(e.norm());
    }
  }
  if (errors.empty()) return;
  svo::robust_cost::MADScaleEstimator scale_estimator;
  // median_of_errors / 1.48
  estimated_scale = scale_estimator.compute(errors);

  num_obs = errors.size();
  chi2_vec_init.reserve(num_obs);
  chi2_vec_final.reserve(num_obs);
  double scale = estimated_scale;
  // 迭代优化
  for (size_t iter = 0; iter < n_iter; iter++) {
    // overwrite scale: 0.85 / focal_length
    // Question: 数学不是很理解
    if (iter == 5) scale = 0.85 / frame->cam_->errorMultiplier2();

    b.setZero();
    A.setZero();
    double new_chi2(0.0);

    // compute residual
    // 遍历所有fts_生成A,b
    for (auto it = frame->fts_.begin(); it != frame->fts_.end(); ++it) {
      if ((*it)->point == NULL) continue;
      Matrix26d J;
      Vector3d xyz_f(frame->T_f_w_ * (*it)->point->pos_);
      // jacobian_xyz2uv计算的就是dx_rn / dT_rn_ro(没有考虑K,或者将f认为等于1)
      Frame::jacobian_xyz2uv(xyz_f, J);

      // z = 1平面上的重投影误差(等价于focal_length = 1)
      Vector2d e = project2d((*it)->f) - project2d(xyz_f);

      double sqrt_inv_cov = 1.0 / (1 << (*it)->level);
      // 重投影误差scale到extracted_level(特征提取时对应的尺度)
      e *= sqrt_inv_cov;

      // 保存优化前的error_vec
      if (iter == 0) {
        if ((*it)->type == Feature::EDGELET) {
          float err_edge = (*it)->grad.transpose() * e;
          chi2_vec_init.push_back(err_edge * err_edge);
        } else
          chi2_vec_init.push_back(e.squaredNorm());  // just for debug
      }

      // 等价于focal_length = sqrt_inv_cov
      J *= sqrt_inv_cov;
      // 目标函数：最小化重投影误差
      // c = Sum(e_i^2)
      // e_i = x_reprojected - x_measured
      // x_projected = [X_reprojected_x / X_reprojected_z,
      //                X_reprojected_y / X_reprojected_z]
      // X_reprojected = Tcw * Xw
      // = K * T_cn_co * T_co_w * Xw
      // = K * T_cn_co * X_co
      // = (I + theta.hat()) X_co + t
      // (此处的重投影误差统一归一化到z=1平面，不考虑f)
      // J = de_i / d(t, theta)
      // = (de_i / dx_reprojected)
      //   * (dx_reprojeced / dX_reprojected)
      //   * (dX_reprojected / d(t, theta))
      // = jacobian_xyz2uv
      // 如果是edgelet，
      // e_i_edgelet = grad.t() * e_i
      // J_edgelet = de_i_edgelet / d(t. theta)
      // = (de_i_edgelet / de_i) * (de_i / d(t, theta))
      // = grad.t() * J
      if ((*it)->type == Feature::EDGELET) {
        Matrix16d J_edge = (*it)->grad.transpose() * J;

        float err_edge = (*it)->grad.transpose() * e;
        // Question: 这种使用robust function的方式和ceres内部有什么区别
        double weight = weight_function.value(std::fabs(err_edge) / scale);

        A.noalias() += J_edge.transpose() * J_edge * weight;
        b.noalias() -= J_edge.transpose() * err_edge * weight;
        new_chi2 += err_edge * err_edge * weight;

      } else {
        double weight = weight_function.value(e.norm() / scale);
        A.noalias() += J.transpose() * J * weight;
        b.noalias() -= J.transpose() * e * weight;
        new_chi2 += e.squaredNorm() * weight;
      }
    }

    // solve linear system
    const Vector6d dT(A.ldlt().solve(b));

    // check if error increased
    if ((iter > 0 && new_chi2 > chi2) || (bool)std::isnan((double)dT[0])) {
      if (verbose)
        std::cout << "it " << iter << "\t FAILURE \t new_chi2 = " << new_chi2
                  << std::endl;
      frame->T_f_w_ = T_old;  // roll-back
      break;
    }

    // update the model
    SE3 T_new = SE3::exp(dT) * frame->T_f_w_;
    // TEST: 下面两句好像顺序反了
    T_old = frame->T_f_w_;
    frame->T_f_w_ = T_new;
    chi2 = new_chi2;
    if (verbose)
      std::cout << "it " << iter << "\t Success \t new_chi2 = " << new_chi2
                << "\t norm(dT) = " << norm_max(dT) << std::endl;

    // stop when converged
    if (norm_max(dT) <= EPS) break;
  }

  // Set covariance as inverse information matrix. Optimistic estimator!
  // 计算优化出的pose的covariance
  // Cov = (A * f^2).inv()
  // 因为A中没有考虑focal_length，所以需要乘以f^2
  const double pixel_variance = 1.0;
  frame->Cov_ = pixel_variance *
                (A * std::pow(frame->cam_->errorMultiplier2(), 2)).inverse();

  // Remove Measurements with too large reprojection error
  // 去除重投影误差较大的点，并且将优化之后的重投影误差保存在chi2_vec_final
  // th_scaled = th / f
  // 因为计算residual时没有考虑focal_length,所以相应的阈值也应该除以f
  double reproj_thresh_scaled = reproj_thresh / frame->cam_->errorMultiplier2();
  size_t n_deleted_refs = 0;
  for (Features::iterator it = frame->fts_.begin(); it != frame->fts_.end();
       ++it) {
    // 跳过无效点
    if ((*it)->point == NULL) continue;

    // 计算重投影误差: z = 1, extracted_level
    Vector2d e =
        project2d((*it)->f) - project2d(frame->T_f_w_ * (*it)->point->pos_);
    double sqrt_inv_cov = 1.0 / (1 << (*it)->level);
    e *= sqrt_inv_cov;

    if ((*it)->type == Feature::EDGELET) {
      float err_edge = (*it)->grad.transpose() * e;
      // BUG: 此处应该是chi2_vec_final
      chi2_vec_init.push_back(err_edge * err_edge);

      if (std::fabs(err_edge) > reproj_thresh_scaled) {
        // we don't need to delete a reference in the point since it was not
        // created yet
        (*it)->point = NULL;
        ++n_deleted_refs;
      }
    } else {
      chi2_vec_final.push_back(e.squaredNorm());
      if (e.norm() > reproj_thresh_scaled) {
        // we don't need to delete a reference in the point since it was not
        // created yet
        (*it)->point = NULL;
        ++n_deleted_refs;
      }
    }
  }

  error_init = 0.0;
  error_final = 0.0;
  // error_init = median_of_chi_vec_init * f
  if (!chi2_vec_init.empty())
    error_init =
        sqrt(getMedian(chi2_vec_init)) * frame->cam_->errorMultiplier2();
  // error_final = median_of_chi_vec_final * f
  if (!chi2_vec_final.empty())
    error_final =
        sqrt(getMedian(chi2_vec_final)) * frame->cam_->errorMultiplier2();

  // estimated_scale *= f
  //（因为重投影误差没有考虑focal_length,所以此处对外输出前需要乘上f）
  estimated_scale *= frame->cam_->errorMultiplier2();
  if (verbose)
    std::cout << "n deleted obs = " << n_deleted_refs
              << "\t scale = " << estimated_scale
              << "\t error init = " << error_init
              << "\t error end = " << error_final << std::endl;
  // num_obs: 重投影误差没有超过阈值的特征数目
  num_obs -= n_deleted_refs;
}
#endif
}  // namespace pose_optimizer
}  // namespace svo
