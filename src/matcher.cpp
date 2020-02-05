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

#include <svo/camera_model.h>
#include <svo/config.h>
#include <svo/feature.h>
#include <svo/feature_alignment.h>
#include <svo/frame.h>
#include <svo/matcher.h>
#include <svo/math_lib.h>
#include <svo/patch_score.h>
#include <svo/point.h>
#include <cstdlib>

namespace svo {

//! Return value between 0 and 255
//! WARNING This function does not check whether the x/y is within the border
// 双线性插值出u,v在mat中的intensity
inline float interpolateMat_8u(const cv::Mat& mat, float u, float v) {
  assert(mat.type() == CV_8U);
  int x = floor(u);
  int y = floor(v);
  float subpix_x = u - x;
  float subpix_y = v - y;

  float w00 = (1.0f - subpix_x) * (1.0f - subpix_y);
  float w01 = (1.0f - subpix_x) * subpix_y;
  float w10 = subpix_x * (1.0f - subpix_y);
  float w11 = 1.0f - w00 - w01 - w10;

  const int stride = mat.step.p[0];
  unsigned char* ptr = mat.data + y * stride + x;
  return w00 * ptr[0] + w01 * ptr[stride] + w10 * ptr[1] +
         w11 * ptr[stride + 1];
}

namespace warp {

void getWarpMatrixAffine(const svo::AbstractCamera& cam_ref,
                         const svo::AbstractCamera& cam_cur,
                         const Vector2d& px_ref, const Vector3d& f_ref,
                         const double depth_ref, const SE3& T_cur_ref,
                         const int level_ref, Matrix2d& A_cur_ref) {
  // Compute affine warp matrix A_cur_ref
  const int halfpatch_size = 5;
  const Vector3d xyz_ref(f_ref * depth_ref);
  Vector3d xyz_du_ref(cam_ref.cam2world(
      px_ref + Vector2d(halfpatch_size, 0) *
                   (1 << level_ref)));  //  patch tranfrom to the level0 pyr img
  Vector3d xyz_dv_ref(cam_ref.cam2world(
      px_ref + Vector2d(0, halfpatch_size) *
                   (1 << level_ref)));  //  px_ref is located at level0
  //  attation!!!! so, A_cur_ref  is only used to affine warp patch at level0
  xyz_du_ref *= xyz_ref[2] / xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2] / xyz_dv_ref[2];
  const Vector2d px_cur(cam_cur.world2cam(T_cur_ref * (xyz_ref)));
  const Vector2d px_du(cam_cur.world2cam(T_cur_ref * (xyz_du_ref)));
  const Vector2d px_dv(cam_cur.world2cam(T_cur_ref * (xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur) / halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur) / halfpatch_size;
}

int getBestSearchLevel(const Matrix2d& A_cur_ref, const int max_level) {
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();
  while (D > 3.0 && search_level < max_level) {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

void warpAffine(const svo::AbstractCamera& cam_ref, const Matrix2d& A_cur_ref,
                const cv::Mat& img_ref, const Vector2d& px_ref,
                const int level_ref, const int search_level,
                const int halfpatch_size, uint8_t* patch) {
  const int patch_size = halfpatch_size * 2;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if (isnan(A_ref_cur(0, 0))) {
    printf("Affine warp is NaN, probably camera has no translation\n");  // TODO
    return;
  }

  // Perform the warp on a larger patch.
  uint8_t* patch_ptr = patch;
  const Vector2f px_ref_pyr = px_ref.cast<float>() / (1 << level_ref);
  for (int y = 0; y < patch_size; ++y) {
    for (int x = 0; x < patch_size; ++x, ++patch_ptr) {
      Vector2f px_patch(x - halfpatch_size, y - halfpatch_size);
      px_patch *= (1 << search_level);
      // const Vector2f px(A_ref_cur*px_patch + px_ref_pyr);
      const Vector2f px(A_cur_ref.cast<float>() * px_patch + px_ref_pyr);
      if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 ||
          px[1] >= img_ref.rows - 1)
        *patch_ptr = 0;
      else {
        // Vector3d temp = cam_ref.cam2world(px[0],px[1]);
        // Vector2d undis_px1 = cam_ref.world2cam( temp );
        Vector2d undis_px = cam_ref.undistortpoint(
            px[0], px[1]);  // we need undistort the image patch
        *patch_ptr =
            (uint8_t)interpolateMat_8u(img_ref, undis_px[0], undis_px[1]);
      }
    }
  }
}

// A_cur_ref: affine warp between cur patch in level0 and ref patch in ref level
// img_ref: ref frame pyramid in ref level
// px_ref: 特征在ref frame中的观测(level 0)
// level_ref: ref level in ref patch
// search_level:　ref_level_in_ref_patch和search_level_in_cur_patch的尺度是一致的（或者说是最接近的）
// halfpatch_size: size of half patch
// patch: ref patch in ref level（是将一个正方形的cur patch in search
// level通过A_cur_ref进行affine warp之后获取的）
void warpAffine(const Matrix2d& A_cur_ref, const cv::Mat& img_ref,
                const Vector2d& px_ref, const int level_ref,
                const int search_level, const int halfpatch_size,
                uint8_t* patch) {
  const int patch_size = halfpatch_size * 2;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if (isnan(A_ref_cur(0, 0))) {
    printf("Affine warp is NaN, probably camera has no translation\n");  // TODO
    return;
  }

  // Perform the warp on a larger patch.
  uint8_t* patch_ptr = patch;
  // px_ref: ref patch in level0, px_ref_pyr: ref patch in ref level
  const Vector2f px_ref_pyr = px_ref.cast<float>() / (1 << level_ref);
  for (int y = 0; y < patch_size; ++y) {
    for (int x = 0; x < patch_size; ++x, ++patch_ptr) {
      // px_patch: cur patch in search_level
      Vector2f px_patch(
          x - halfpatch_size,
          y - halfpatch_size);  // px_patch is locat at  pyr [ref_level ]
      // px_patch: cur patch in level0
      px_patch *= (1 << search_level);  //  1. patch tranform to level0,
                                        //  because A_ref_cur is only used to
                                        //  affine warp level0 patch
      // A_ref_cur是ref patch in ref level和cur patch in level0之间的affine warp
      // px: ref patch in ref level(和cur patch in search
      // level中的pixel一一对应)
      const Vector2f px(
          A_ref_cur * px_patch +
          px_ref_pyr);  //  2. then, use A_ref_cur  to affine warp the patch
      if (px[0] < 0 || px[1] < 0 || px[0] >= img_ref.cols - 1 ||
          px[1] >= img_ref.rows - 1)
        *patch_ptr = 0;
      else {
        // 获取ref frame pyramid in ref level中在px处的双线性插值的intensity
        *patch_ptr = (uint8_t)interpolateMat_8u(
            img_ref, px[0], px[1]);  // img_ref  is the  img at pyr[level]
      }
    }
  }
}

}  // namespace warp

bool depthFromTriangulation(const SE3& T_search_ref, const Vector3d& f_ref,
                            const Vector3d& f_cur, double& depth) {
  Matrix<double, 3, 2> A;
  A << T_search_ref.rotation_matrix() * f_ref, f_cur;
  const Matrix2d AtA = A.transpose() * A;
  if (AtA.determinant() < 0.000001) return false;
  const Vector2d depth2 =
      -AtA.inverse() * A.transpose() * T_search_ref.translation();
  depth = fabs(depth2[0]);
  return true;
}

// 从patch_with_border_抽取其中心的patch存储到patch_
void Matcher::createPatchFromPatchWithBorder() {
  uint8_t* ref_patch_ptr = patch_;
  for (int y = 1; y < patch_size_ + 1; ++y, ref_patch_ptr += patch_size_) {
    uint8_t* ref_patch_border_ptr =
        patch_with_border_ + y * (patch_size_ + 2) + 1;
    for (int x = 0; x < patch_size_; ++x)
      ref_patch_ptr[x] = ref_patch_border_ptr[x];
  }
}

// pt: 待匹配的特征的Point信息
// cur_frame: 当前帧信息
// px_cur: 初值表示待匹配特征在某个关键帧上的观测，最终会收敛到当前帧观测的位置
bool Matcher::findMatchDirect(const Point& pt, const Frame& cur_frame,
                              Vector2d& px_cur) {
  // 查找当前特征的所有关键帧观测中和当前帧观测视角最接近的关键帧观测(ref_ftr_)
  if (!pt.getCloseViewObs(cur_frame.pos(), ref_ftr_)) {
    // std::cout<< "\033[1;32m"<<" can not getCloseViewObs!"<<" \033[0m"
    // <<std::endl;
    return false;
  }

  // 判断找出的ref_ftr_是否在该关键帧的指定层金字塔的中心区域
  if (!ref_ftr_->frame->cam_->isInFrame(
          ref_ftr_->px.cast<int>() / (1 << ref_ftr_->level),
          halfpatch_size_ + 2, ref_ftr_->level)) {
    // std::cout<< "\033[1;32m"<<" not in frame!"<<" \033[0m" <<std::endl;
    return false;
  }

  // warp affine: 获取cur frame和ref frame之间的affine warp(A_cur_ref_: 2x2)
  warp::getWarpMatrixAffine(
      *ref_ftr_->frame->cam_, *cur_frame.cam_, ref_ftr_->px, ref_ftr_->f,
      (ref_ftr_->frame->pos() - pt.pos_).norm(),
      cur_frame.T_f_w_ * ref_ftr_->frame->T_f_w_.inverse(), ref_ftr_->level,
      A_cur_ref_);
  // 找出cur frame和ref frame feature尺度最接近的level（cur frame）
  search_level_ =
      warp::getBestSearchLevel(A_cur_ref_, Config::nPyrLevels() - 1);
  // 将一个正方形的cur patch in search level通过A_cur_ref进行affine
  // warp之后获取ref patch in ref level (patch_with_border_)
  warp::warpAffine(A_cur_ref_, ref_ftr_->frame->img_pyr_[ref_ftr_->level],
                   ref_ftr_->px, ref_ftr_->level, search_level_,
                   halfpatch_size_ + 1, patch_with_border_);
  // 从patch_with_border_抽取其中心的patch存储到patch_
  createPatchFromPatchWithBorder();

  // px_cur should be set
  // 根据某个关键帧观测来设置当前帧观测的初值并且scale到search_level_
  Vector2d px_scaled(px_cur / (1 << search_level_));

  // 根据feature type类型分别采取不同的align方式
  bool success = false;
  if (ref_ftr_->type == Feature::EDGELET) {
    Vector2d dir_cur(A_cur_ref_ * ref_ftr_->grad);
    dir_cur.normalize();
    // input说明：
    // cur_frame.img_pyr_[search_level_]: cur frame pyramid in search_level_
    // dir_cur: ref frame的grad affine warp到cur
    // frame(由于grad不随着level而变化，所以不考虑scale的问题)
    // patch_with_border_: 将一个正方形的cur patch in search
    // level通过A_cur_ref进行affine warp之后获取ref patch in ref level
    // patch_: patch_with_border_的中心区域
    // options_.align_max_iter: 10
    // px_scaled: 初值是某个关键帧观测scale到search level，输出是跟踪之后的结果
    // Question: h_inv_: ???
    success = feature_alignment::align1D(
        cur_frame.img_pyr_[search_level_], dir_cur.cast<float>(),
        patch_with_border_, patch_, options_.align_max_iter, px_scaled, h_inv_);
  } else {
    success = feature_alignment::align2D(cur_frame.img_pyr_[search_level_],
                                         patch_with_border_, patch_,
                                         options_.align_max_iter, px_scaled);
  }

  // 将跟踪结果scale到最fine的那一层
  px_cur = px_scaled * (1 << search_level_);

  // 返回align是否成功的标志位
  return success;
}

bool Matcher::findEpipolarMatchDirect(const Frame& ref_frame,
                                      const Frame& cur_frame,
                                      const Feature& ref_ftr,
                                      const double d_estimate,
                                      const double d_min, const double d_max,
                                      double& depth) {
  SE3 T_cur_ref = cur_frame.T_f_w_ * ref_frame.T_f_w_.inverse();
  int zmssd_best = PatchScore::threshold();
  Vector2d uv_best;

  // Compute start and end of epipolar line in old_kf for match search, on unit
  // plane!
  Vector2d A = project2d(T_cur_ref * (ref_ftr.f * d_min));
  Vector2d B = project2d(T_cur_ref * (ref_ftr.f * d_max));
  epi_dir_ = A - B;

  // Compute affine warp matrix
  warp::getWarpMatrixAffine(*ref_frame.cam_, *cur_frame.cam_, ref_ftr.px,
                            ref_ftr.f, d_estimate, T_cur_ref, ref_ftr.level,
                            A_cur_ref_);

  // feature pre-selection
  reject_ = false;
  if (ref_ftr.type == Feature::EDGELET &&
      options_.epi_search_edgelet_filtering) {
    const Vector2d grad_cur = (A_cur_ref_ * ref_ftr.grad).normalized();
    const double cosangle = fabs(grad_cur.dot(epi_dir_.normalized()));
    if (cosangle < options_.epi_search_edgelet_max_angle) {
      reject_ = true;
      return false;
    }
  }

  search_level_ =
      warp::getBestSearchLevel(A_cur_ref_, Config::nPyrLevels() - 1);

  // Find length of search range on epipolar line
  Vector2d px_A(cur_frame.cam_->world2cam(A));
  Vector2d px_B(cur_frame.cam_->world2cam(B));
  epi_length_ = (px_A - px_B).norm() / (1 << search_level_);

  // Warp reference patch at ref_level

  warp::warpAffine(A_cur_ref_, ref_frame.img_pyr_[ref_ftr.level], ref_ftr.px,
                   ref_ftr.level, search_level_, halfpatch_size_ + 1,
                   patch_with_border_);
  /*
  warp::warpAffine(*ref_frame.cam_,A_cur_ref_,
  ref_frame.img_pyr_[ref_ftr.level], ref_ftr.px,
                   ref_ftr.level, search_level_, halfpatch_size_+1,
  patch_with_border_);
*/
  createPatchFromPatchWithBorder();

  if (epi_length_ < 2.0) {
    px_cur_ = (px_A + px_B) / 2.0;
    Vector2d px_scaled(px_cur_ / (1 << search_level_));
    bool res;
    if (options_.align_1d)
      res = feature_alignment::align1D(
          cur_frame.img_pyr_[search_level_],
          (px_A - px_B).cast<float>().normalized(), patch_with_border_, patch_,
          options_.align_max_iter, px_scaled, h_inv_);
    else
      res = feature_alignment::align2D(cur_frame.img_pyr_[search_level_],
                                       patch_with_border_, patch_,
                                       options_.align_max_iter, px_scaled);
    if (res) {
      px_cur_ = px_scaled * (1 << search_level_);
      if (depthFromTriangulation(T_cur_ref, ref_ftr.f,
                                 cur_frame.cam_->cam2world(px_cur_), depth))
        return true;
    }
    return false;
  }

  size_t n_steps = epi_length_ / 0.7;  // one step per pixel
  Vector2d step = epi_dir_ / n_steps;

  if (n_steps > options_.max_epi_search_steps) {
    printf(
        "WARNING: skip epipolar search: %zu evaluations, px_lenght=%f, "
        "d_min=%f, d_max=%f.\n",
        n_steps, epi_length_, d_min, d_max);
    return false;
  }

  // for matching, precompute sum and sum2 of warped reference patch
  int pixel_sum = 0;
  int pixel_sum_square = 0;
  PatchScore patch_score(patch_);

  // now we sample along the epipolar line
  Vector2d uv = B - step;
  Vector2i last_checked_pxi(0, 0);
  ++n_steps;
  for (size_t i = 0; i < n_steps; ++i, uv += step) {
    Vector2d px(cur_frame.cam_->world2cam(uv));
    Vector2i pxi(
        px[0] / (1 << search_level_) + 0.5,
        px[1] / (1 << search_level_) + 0.5);  // +0.5 to round to closest int

    if (pxi == last_checked_pxi) continue;
    last_checked_pxi = pxi;

    // check if the patch is full within the new frame
    if (!cur_frame.cam_->isInFrame(pxi, patch_size_, search_level_)) continue;

    // TODO interpolation would probably be a good idea
    uint8_t* cur_patch_ptr =
        cur_frame.img_pyr_[search_level_].data +
        (pxi[1] - halfpatch_size_) * cur_frame.img_pyr_[search_level_].cols +
        (pxi[0] - halfpatch_size_);
    int zmssd = patch_score.computeScore(
        cur_patch_ptr, cur_frame.img_pyr_[search_level_].cols);

    if (zmssd < zmssd_best) {
      zmssd_best = zmssd;
      uv_best = uv;
    }
  }

  if (zmssd_best < PatchScore::threshold()) {
    if (options_.subpix_refinement) {
      px_cur_ = cur_frame.cam_->world2cam(uv_best);
      Vector2d px_scaled(px_cur_ / (1 << search_level_));
      bool res;
      if (options_.align_1d)
        res = feature_alignment::align1D(
            cur_frame.img_pyr_[search_level_],
            (px_A - px_B).cast<float>().normalized(), patch_with_border_,
            patch_, options_.align_max_iter, px_scaled, h_inv_);
      else
        res = feature_alignment::align2D(cur_frame.img_pyr_[search_level_],
                                         patch_with_border_, patch_,
                                         options_.align_max_iter, px_scaled);
      if (res) {
        px_cur_ = px_scaled * (1 << search_level_);
        if (depthFromTriangulation(T_cur_ref, ref_ftr.f,
                                   cur_frame.cam_->cam2world(px_cur_), depth))
          return true;
      }
      return false;
    }
    px_cur_ = cur_frame.cam_->world2cam(uv_best);
    if (depthFromTriangulation(T_cur_ref, ref_ftr.f,
                               unproject2d(uv_best).normalized(), depth))
      return true;
  }
  return false;
}

}  // namespace svo
