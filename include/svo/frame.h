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

#ifndef SVO_FRAME_H_
#define SVO_FRAME_H_

#include <sophus/se3.h>
#include <svo/camera_model.h>
#include <svo/global.h>
#include <svo/math_lib.h>
#include <boost/noncopyable.hpp>

namespace g2o {
class VertexSE3Expmap;
}
typedef g2o::VertexSE3Expmap g2oFrameSE3;

namespace svo {

class Point;
struct Feature;

typedef list<Feature*> Features;
typedef vector<cv::Mat> ImgPyr;

/// A frame saves the image, the associated features and the estimated pose.
class Frame : boost::noncopyable {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static int frame_counter_;  //!< Counts the number of created frames. Used to
                              //! set the unique id.
  int id_;                    //!< Unique id of the frame.
  double timestamp_;          //!< Timestamp of when the image was recorded.
  svo::AbstractCamera* cam_;  //!< Camera model.
  Sophus::SE3 T_f_w_;         //!< Transform (f)rame from (w)orld.
  // Question: 优化的方案是怎么计算Cov的？
  Matrix<double, 6, 6> Cov_;  //!< Covariance.
  cv::Mat debug_img_;         // used to draw feature in img_pyr_[0]
  ImgPyr img_pyr_;            //!< Image Pyramid.
  // 保存了3d vs 2d alginment（reprojectMap）中成功跟踪的3d feature
  Features fts_;  //!< List of features in the image.
  // 为什么只用5个点计算overlap: 5个点分别是最边界的4个和最中心的一个
  vector<Feature*> key_pts_;  //!< Five features and associated 3D points which
                              //! are used to detect if two frames have
  //! overlapping field of view.
  // Question: 关键帧判断逻辑是怎样的?
  bool is_keyframe_;  //!< Was this frames selected as keyframe?
  // 是否已经执行过DepthFilter::initializeSeeds
  bool have_initializeSeeds;
  g2oFrameSE3*
      v_kf_;  //!< Temporary pointer to the g2o node object of the keyframe.
  int last_published_ts_;  //!< Timestamp of last publishing.

  Frame(svo::AbstractCamera* cam, const cv::Mat& img, double timestamp);
  ~Frame();

  /// Initialize new frame and create image pyramid.
  void initFrame(const cv::Mat& img);

  /// Select this frame as keyframe.
  void setKeyframe();

  /// Add a feature to the image
  void addFeature(Feature* ftr);

  /// The KeyPoints are those five features which are closest to the 4 image
  /// corners
  /// and to the center and which have a 3D point assigned. These points are
  /// used
  /// to quickly check whether two frames have overlapping field of view.
  void setKeyPoints();

  /// Check if we can select five better key-points.
  void checkKeyPoints(Feature* ftr);

  /// If a point is deleted, we must remove the corresponding key-point.
  void removeKeyPoint(Feature* ftr);

  /// Return number of point observations.
  inline size_t nObs() const { return fts_.size(); }

  /// Check if a point in (w)orld coordinate frame is visible in the image.
  bool isVisible(const Vector3d& xyz_w) const;

  /// Full resolution image stored in the frame.
  inline const cv::Mat& img() const { return img_pyr_[0]; }

  /// Was this frame selected as keyframe?
  inline bool isKeyframe() const { return is_keyframe_; }

  /// Transforms point coordinates in world-frame (w) to camera pixel
  /// coordinates (c).
  inline Vector2d w2c(const Vector3d& xyz_w) const {
    return cam_->world2cam(T_f_w_ * xyz_w);
  }

  /// Transforms pixel coordinates (c) to frame unit sphere coordinates (f).
  inline Vector3d c2f(const Vector2d& px) const {
    return cam_->cam2world(px[0], px[1]);
  }

  /// Transforms pixel coorfuhaodinates (c) to frame unit sphere coordinates
  /// (f).
  inline Vector3d c2f(const double x, const double y) const {
    return cam_->cam2world(x, y);
  }

  /// Transforms point coordinates in world-frame (w) to camera-frams (f).
  inline Vector3d w2f(const Vector3d& xyz_w) const { return T_f_w_ * xyz_w; }

  /// Transforms point from frame unit sphere (f) frame to world coordinate
  /// frame (w).
  inline Vector3d f2w(const Vector3d& f) const { return T_f_w_.inverse() * f; }

  /// Projects Point from unit sphere (f) in camera pixels (c).
  inline Vector2d f2c(const Vector3d& f) const { return cam_->world2cam(f); }

  /// Return the pose of the frame in the (w)orld coordinate frame.
  inline Vector3d pos() const { return T_f_w_.inverse().translation(); }

  /// Frame jacobian for projection of 3D point in (f)rame coordinate to
  /// unit plane coordinates uv (focal length = 1).
  // 推导如下：
  // c = sum_of_all_pixels (T((K * T_rn_ro * X_ro)_前两维) - I(x_c))^2
  // 注：　
  // 1. r: reference frame, n: new, o: old, c: current
  // 2. 为了提升计算速度，使用的是inverse compositional method
  // (即变化量发生在reference frame)
  // 3. 对T进行求导： dT / dT_rn_ro = (dT / dx_rn) * (dx_rn / dT_rn_ro)
  // jacobian_xyz2uv计算的就是dx_rn / dT_rn_ro(没有考虑K,或者将f认为等于1)
  // 下面以u为例推导：
  // 正向推导过程：
  // X_rn = T_rn_ro * X_ro = (I + theta.hat()) * X_ro + t
  // x_rn_x = X_rn_x / X_rn_z
  // x_rn_u = f * x_rn_x + cu
  // 链式求导过程：
  // dx_rn_u / d(t, theta) =
  // (dx_rn_u / d_x_rn_x) * (dx_rn_x / dX_rn) * (dX_rn / d(t, theta))
  // dx_rn_u / d_x_rn_x = f
  // dx_rn_x / dX_rn = (1 / X_rn_z, 0, -X_rn_x / X_rn_z^2)
  // dX_rn / dt = I; dX_rn / dtheta = X_ro.hat()
  // Question:
  // 1. 为什么jacobian乘以了-1
  // 2. 实现中是否做了如下近似: X_rn = X_ro
  inline static void jacobian_xyz2uv(const Vector3d& xyz_in_f,
                                     Matrix<double, 2, 6>& J) {
    const double x = xyz_in_f[0];
    const double y = xyz_in_f[1];
    const double z_inv = 1. / xyz_in_f[2];
    const double z_inv_2 = z_inv * z_inv;

    J(0, 0) = -z_inv;                // -1/z
    J(0, 1) = 0.0;                   // 0
    J(0, 2) = x * z_inv_2;           // x/z^2
    J(0, 3) = y * J(0, 2);           // x*y/z^2
    J(0, 4) = -(1.0 + x * J(0, 2));  // -(1.0 + x^2/z^2)
    J(0, 5) = y * z_inv;             // y/z

    J(1, 0) = 0.0;                // 0
    J(1, 1) = -z_inv;             // -1/z
    J(1, 2) = y * z_inv_2;        // y/z^2
    J(1, 3) = 1.0 + y * J(1, 2);  // 1.0 + y^2/z^2
    J(1, 4) = -J(0, 3);           // -x*y/z^2
    J(1, 5) = -x * z_inv;         // -x/z
  }
};

/// Some helper functions for the frame object.
namespace frame_utils {

/// Creates an image pyramid of half-sampled images.
void createImgPyramid(const cv::Mat& img_level_0, int n_levels, ImgPyr& pyr);

/// Get the average depth of the features in the image.
bool getSceneDepth(const Frame& frame, double& depth_mean, double& depth_min);

}  // namespace frame_utils
}  // namespace svo

#endif  // SVO_FRAME_H_
