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

//#include <vikit/abstract_camera.h>
#include <stdlib.h>
#include <svo/config.h>
#include <svo/feature.h>
#include <svo/frame_handler_base.h>
#include <svo/map.h>
#include <svo/matcher.h>
#include <svo/point.h>
#include <Eigen/StdVector>
#include <boost/bind.hpp>
#include <fstream>

namespace svo {

// definition of global and static variables which were declared in the header
#ifdef SVO_TRACE
svo::DeBuger* g_permon = NULL;
#endif

FrameHandlerBase::FrameHandlerBase()
    : stage_(STAGE_PAUSED),
      set_reset_(false),
      set_start_(false),
      acc_frame_timings_(10),
      acc_num_obs_(10),
      num_obs_last_(0),
      tracking_quality_(TRACKING_INSUFFICIENT) {
#ifdef SVO_TRACE
  // Initialize Performance Monitor
  g_permon = new svo::DeBuger();
  g_permon->addTimer("pyramid_creation");
  g_permon->addTimer("sparse_img_align");
  g_permon->addTimer("reproject");
  g_permon->addTimer("reproject_kfs");
  g_permon->addTimer("reproject_candidates");
  g_permon->addTimer("feature_align");
  g_permon->addTimer("pose_optimizer");
  g_permon->addTimer("point_optimizer");
  g_permon->addTimer("local_ba");
  g_permon->addTimer("tot_time");
  g_permon->addLog("timestamp");
  g_permon->addLog("img_align_n_tracked");
  g_permon->addLog("repr_n_mps");
  g_permon->addLog("repr_n_new_references");
  g_permon->addLog("sfba_thresh");
  g_permon->addLog("sfba_error_init");
  g_permon->addLog("sfba_error_final");
  g_permon->addLog("sfba_n_edges_final");
  g_permon->addLog("loba_n_erredges_init");
  g_permon->addLog("loba_n_erredges_fin");
  g_permon->addLog("loba_err_init");
  g_permon->addLog("loba_err_fin");
  g_permon->addLog("n_candidates");
  g_permon->addLog("dropout");
  g_permon->init(Config::traceName(), Config::traceDir());
#endif

  SVO_INFO_STREAM("SVO initialized");
}

FrameHandlerBase::~FrameHandlerBase() {
  SVO_INFO_STREAM("SVO destructor invoked");
#ifdef SVO_TRACE
  delete g_permon;
#endif
}

// 一些辅助工作:
// 1. 全局第一帧的初始化逻辑(状态管理)
// 2. 配置计时功能
// 3. 清空map_
bool FrameHandlerBase::startFrameProcessingCommon(const double timestamp) {
  if (set_start_) {
    resetAll();
    stage_ = STAGE_FIRST_FRAME;
  }

  if (stage_ == STAGE_PAUSED) return false;

  SVO_LOG(timestamp);
  SVO_DEBUG_STREAM("New Frame");
  SVO_START_TIMER("tot_time");
  timer_.start();

  // some cleanup from last iteration, can't do before because of visualization
  map_.emptyTrash();
  return true;
}

// 主要工作:
// 1. 打印log
// 2. 统计耗时
// 3. 状态管理
int FrameHandlerBase::finishFrameProcessingCommon(
    const size_t update_id, const UpdateResult dropout,
    const size_t num_observations) {
  SVO_DEBUG_STREAM("Frame: " << update_id << "\t fps-avg = "
                             << 1.0 / acc_frame_timings_.getMean()
                             << "\t nObs = " << acc_num_obs_.getMean());
  SVO_LOG(dropout);

  // save processing time to calculate fps
  acc_frame_timings_.push_back(timer_.stop());
  if (stage_ == STAGE_DEFAULT_FRAME) acc_num_obs_.push_back(num_observations);
  num_obs_last_ = num_observations;
  SVO_STOP_TIMER("tot_time");

  // Question: 没有看懂
#ifdef SVO_TRACE
  g_permon->writeToFile();
  {
    boost::unique_lock<boost::mutex> lock(map_.point_candidates_.mut_);
    size_t n_candidates = map_.point_candidates_.candidates_.size();
    SVO_LOG(n_candidates);
  }
#endif

  // 状态管理：根据当前帧处理状态，决定整个程序的状态
  if (dropout == RESULT_FAILURE &&
      (stage_ == STAGE_DEFAULT_FRAME || stage_ == STAGE_RELOCALIZING)) {
    stage_ = STAGE_RELOCALIZING;
    tracking_quality_ = TRACKING_INSUFFICIENT;
  } else if (dropout == RESULT_FAILURE)
    resetAll();
  if (set_reset_) resetAll();

  return 0;
}

void FrameHandlerBase::resetCommon() {
  map_.reset();
  stage_ = STAGE_PAUSED;
  set_reset_ = false;
  set_start_ = false;
  tracking_quality_ = TRACKING_INSUFFICIENT;
  num_obs_last_ = 0;
  SVO_INFO_STREAM("RESET");
}

// num_observations: 在optimizeGaussNewton之后重投影误差小于阈值的特征数目
// init_match_number: 当前帧在reprojectMap中跟踪成功的3d特征
// 根据optimizeGaussNewton和reprojectMap操作之后”存活“的特征数目决定tracking_quality_
void FrameHandlerBase::setTrackingQuality(const size_t num_observations,
                                          const size_t init_match_number) {
  tracking_quality_ = TRACKING_GOOD;
  if (num_observations < Config::qualityMinFts()) {
    SVO_WARN_STREAM_THROTTLE(
        0.5, "Tracking less than " << Config::qualityMinFts() << " features!");
    tracking_quality_ = TRACKING_INSUFFICIENT;
  }

  // const int feature_drop = static_cast<int>(std::min(num_obs_last_,
  // Config::maxFts())) - num_observations;
  // if(feature_drop > Config::qualityMaxFtsDrop())
  const float feature_drop = (float)(init_match_number - num_observations);
  if (feature_drop / init_match_number > 0.6) {
    // SVO_WARN_STREAM("outliter during pose  "<< feature_drop <<" features!");
    SVO_WARN_STREAM("outliter: "
                    << feature_drop
                    << " init_match_number: " << init_match_number << " "
                    << feature_drop * 1.0 / init_match_number * 100.0 << "%");
    tracking_quality_ = TRACKING_INSUFFICIENT;
  }
}

bool ptLastOptimComparator(Point* lhs, Point* rhs) {
  return (lhs->last_structure_optim_ < rhs->last_structure_optim_);
}

// frame: current_frame_
// max_n_pts: Maximum number of points to optimize at every iteration.
// max_iter: Number of iterations in structure optimization
void FrameHandlerBase::optimizeStructure(FramePtr frame, size_t max_n_pts,
                                         int max_iter) {
  // 获取当前帧所有跟踪成功并且优化之后重投影误差小于阈值的特征(pts)
  deque<Point*> pts;
  for (Features::iterator it = frame->fts_.begin(); it != frame->fts_.end();
       ++it) {
    if ((*it)->point != NULL) pts.push_back((*it)->point);
  }
  max_n_pts = min(max_n_pts, pts.size());
  // 对前max_n_pts个特征进行排序
  //（按照特征上次进行optimizeStructure操作的时间戳排序,实际实现中使用的新增观测的id）
  nth_element(pts.begin(), pts.begin() + max_n_pts, pts.end(),
              ptLastOptimComparator);
  // 遍历pts对每个特征进行3d点优化，并且更新last_structure_optim_
  for (deque<Point*>::iterator it = pts.begin(); it != pts.begin() + max_n_pts;
       ++it) {
    (*it)->optimize(max_iter);
    (*it)->last_structure_optim_ = frame->id_;
  }
}

}  // namespace svo
