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

#include <svo/config.h>
#include <svo/depth_filter.h>
#include <svo/feature.h>
#include <svo/feature_detection.h>
#include <svo/frame.h>
#include <svo/global.h>
#include <svo/matcher.h>
#include <svo/point.h>
#include <algorithm>
#include <boost/bind.hpp>
#include <boost/math/distributions/normal.hpp>

namespace svo {

int Seed::batch_counter = 0;
int Seed::seed_counter = 0;

Seed::Seed(Feature* ftr, float depth_mean, float depth_min)
    : batch_id(batch_counter),
      id(seed_counter++),
      ftr(ftr),
      a(10),
      b(10),
      mu(1.0 / depth_mean),
      z_range(1.0 / depth_min),
      sigma2(z_range * z_range / 36)  // 36
{}

DepthFilter::DepthFilter(feature_detection::DetectorPtr feature_detector,
                         feature_detection::DetectorPtr edge_detector,
                         callback_t seed_converged_cb)
    : feature_detector_(feature_detector),
      edge_detector_(edge_detector),
      seed_converged_cb_(seed_converged_cb),
      seeds_updating_halt_(false),
      thread_(NULL),
      new_keyframe_set_(false),
      new_keyframe_min_depth_(0.0),
      new_keyframe_mean_depth_(0.0) {}

DepthFilter::~DepthFilter() {
  stopThread();
  SVO_INFO_STREAM("DepthFilter destructed.");
}

void DepthFilter::startThread() {
  thread_ = new boost::thread(&DepthFilter::updateSeedsLoop, this);
}

void DepthFilter::stopThread() {
  SVO_INFO_STREAM("DepthFilter stop thread invoked.");
  if (thread_ != NULL) {
    SVO_INFO_STREAM("DepthFilter interrupt and join thread... ");
    seeds_updating_halt_ = true;
    thread_->interrupt();
    thread_->join();
    thread_ = NULL;
  }
}

void DepthFilter::addFrame(FramePtr frame) {
  if (thread_ != NULL) {
    {
      lock_t lock(frame_queue_mut_);
      if (frame_queue_.size() > 5) frame_queue_.pop();
      frame_queue_.push(frame);
    }
    seeds_updating_halt_ = false;
    frame_queue_cond_.notify_one();
  } else
    updateSeeds(frame);
}

void DepthFilter::addKeyframe(FramePtr frame, double depth_mean,
                              double depth_min) {
  new_keyframe_min_depth_ = depth_min;
  new_keyframe_mean_depth_ = depth_mean;
  if (thread_ != NULL) {
    new_keyframe_ = frame;
    new_keyframe_set_ = true;
    seeds_updating_halt_ = true;
    frame_queue_cond_.notify_one();
  } else
    initializeSeeds(frame);
}

// 检测新特征并且为新特征初始化depth filter(Seed)
void DepthFilter::initializeSeeds(FramePtr frame) {
  // 如果frame已经执行过DepthFilter::initializeSeeds，则直接返回，
  // 确保每个frame只执行一次DepthFilter::initializeSeeds
  if (frame->have_initializeSeeds) return;

  // TEST: 尝试只使用edge或者corner的效果
  // 新检测出的corner和edge特征
  Features new_features;
  // 将已检测出特征的grid设置为occupied，不再检测特征
  feature_detector_->setExistingFeatures(frame->fts_);
  // 在frame中non-occpupied的grid中检测新的corner特征
  feature_detector_->detect(frame.get(), frame->img_pyr_,
                            Config::triangMinCornerScore(), new_features);
  // 检测edgelet特征
  edge_detector_->detect(frame.get(), frame->img_pyr_,
                         Config::triangMinCornerScore(), new_features);

  // initialize a seed for every new feature
  // 暂停seeds update
  seeds_updating_halt_ = true;
  lock_t lock(seeds_mut_);  // by locking the updateSeeds function stops
  // new Seed之前batch_counter自增1
  ++Seed::batch_counter;

  // SVO_DEBUG_STREAM("67676767676.");
  // 利用相邻相似原理为每个新检测的特征初始化depth,以及new Seed加入到seeds_中
  std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr) {
    // seeds_.push_back(Seed(ftr, new_keyframe_mean_depth_,
    // new_keyframe_min_depth_));

    // hyj  fix code : useing neighbouring fts to init the seeds depth
    // 找出离当前新检测特征x最临近（image
    // plane）的跟踪成功的特征y，并且x.depth=y.depth
    double dist_min(100), z(new_keyframe_mean_depth_);
    for (auto it = frame->fts_.begin(), ite = frame->fts_.end(); it != ite;
         ++it) {
      if ((*it)->point != NULL) {
        // 新检测特征和跟踪成功特征之间的距离
        Vector2d dist = (ftr->px - (*it)->px);
        if (dist.norm() < dist_min) {
          dist_min = dist.norm();
          z = frame->w2f((*it)->point->pos_).z();
        }
      }
    }
    if (dist_min < 70)
      seeds_.push_back(Seed(ftr, z, new_keyframe_min_depth_));
    else
      // 异常处理:
      // 如果没有比较临近的特征，mean_depth就设置为new_keyframe_mean_depth_
      seeds_.push_back(
          Seed(ftr, new_keyframe_mean_depth_, new_keyframe_min_depth_));
  });

  // SVO_DEBUG_STREAM("68686868686.");
  if (options_.verbose)
    SVO_INFO_STREAM("DepthFilter: Initialized " << new_features.size()
                                                << " new seeds");
  // 打开seeds update
  seeds_updating_halt_ = false;

  frame->have_initializeSeeds = true;
}

// 删除seeds_和frame相关联的Seed
// frame: 需要被删除的关键帧
void DepthFilter::removeKeyframe(FramePtr frame) {
  // 删除关键帧之前需要先将seed update暂停
  seeds_updating_halt_ = true;
  lock_t lock(seeds_mut_);
  list<Seed>::iterator it = seeds_.begin();
  size_t n_removed = 0;
  // SVO_DEBUG_STREAM("6666666666666666.");
  while (it != seeds_.end()) {
    if (it->ftr->frame == frame.get()) {
      it = seeds_.erase(it);
      ++n_removed;
    } else
      ++it;
  }
  // SVO_DEBUG_STREAM("77777777777.");
  // 删除关键帧观测之后还需要将seed update重启
  seeds_updating_halt_ = false;
}

void DepthFilter::reset() {
  seeds_updating_halt_ = true;
  {
    lock_t lock(seeds_mut_);
    seeds_.clear();
  }
  lock_t lock();
  while (!frame_queue_.empty()) frame_queue_.pop();
  seeds_updating_halt_ = false;

  if (options_.verbose) SVO_INFO_STREAM("DepthFilter: RESET.");
}

void DepthFilter::updateSeedsLoop() {
  while (!boost::this_thread::interruption_requested()) {
    FramePtr frame;
    {
      lock_t lock(frame_queue_mut_);
      while (frame_queue_.empty() && new_keyframe_set_ == false)
        frame_queue_cond_.wait(lock);
      if (new_keyframe_set_) {
        new_keyframe_set_ = false;
        seeds_updating_halt_ = false;
        // clearFrameQueue();
        frame = new_keyframe_;
      } else {
        frame = frame_queue_.front();
        frame_queue_.pop();
      }
    }
    updateSeeds(frame);
    if (frame->isKeyframe()) initializeSeeds(frame);
  }
}

void DepthFilter::updateSeeds(FramePtr frame) {
  // update only a limited number of seeds, because we don't have time to do it
  // for all the seeds in every frame!
  size_t n_updates = 0, n_failed_matches = 0, n_seeds = seeds_.size();
  lock_t lock(seeds_mut_);
  list<Seed>::iterator it = seeds_.begin();

  const double focal_length = frame->cam_->errorMultiplier2();
  double px_noise = 1.0;
  // Question: law of chord
  double px_error_angle =
      atan(px_noise / (2.0 * focal_length)) * 2.0;  // law of chord (sehnensatz)

  // SVO_DEBUG_STREAM("55555555555555.");
  int erase_seed = 0;
  while (it != seeds_.end()) {
    // set this value true when seeds updating should be interrupted
    if (seeds_updating_halt_) return;

    // check if seed is not already too old
    // 近似等于该Seed对象从生成到当前总共经历了几个关键帧（即Seed对象的存活时间）
    if ((Seed::batch_counter - it->batch_id) > options_.max_n_kfs) {
      // 如果seed太老，直接删除
      it = seeds_.erase(it);
      continue;
    }
    /*
        if(( it->sigma2 > (it->z_range*it->z_range)/25 ))// || ( (it->mu -
       sqrt(it->sigma2))<0 ) )
        {
          //std::cout<<"sigma2 is larger than z_range"<<std::endl;
          it = seeds_.erase(it);
          erase_seed ++;
          continue;
        }
    */
    // check if point is visible in the current image
    SE3 T_ref_cur = it->ftr->frame->T_f_w_ * frame->T_f_w_.inverse();
    // 根据depth filter估计的inverse depth以及观测的bearing
    // vector和帧间位姿获取当前帧3d点坐标
    const Vector3d xyz_f(T_ref_cur.inverse() * (1.0 / it->mu * it->ftr->f));
    if (xyz_f.z() < 0.0) {
      ++it;  // behind the camera
      continue;
    }
    if (!frame->cam_->isInFrame(frame->f2c(xyz_f).cast<int>())) {
      ++it;  // point does not project in image
      continue;
    }

    // we are using inverse depth coordinates
    float sigma = sqrt(it->sigma2);
    float z_inv_min = it->mu + sigma;
    float z_inv_max = max(it->mu - sigma, 0.00000001f);
    double z;
    if (!matcher_.findEpipolarMatchDirect(*it->ftr->frame, *frame, *it->ftr,
                                          1.0 / it->mu, 1.0 / z_inv_min,
                                          1.0 / z_inv_max, z)) {
      it->b++;  // increase outlier probability when no match was found
      ++it;
      ++n_failed_matches;
      continue;
    }

    // compute tau
    double tau = computeTau(T_ref_cur, it->ftr->f, z, px_error_angle);
    // 1pixel能够导致的最大的逆深度误差(0.5是因为括号中是最大逆深度和最小逆深度的插值)
    double tau_inverse =
        0.5 * (1.0 / max(0.0000001, z - tau) - 1.0 / (z + tau));

    // update the estimate
    updateSeed(1. / z, tau_inverse * tau_inverse, &*it);
    ++n_updates;

    // Question: 这一策略在svo具体如何起作用还没有理清楚
    if (frame->isKeyframe()) {
      // The feature detector should not initialize new seeds close to this
      // location
      feature_detector_->setGridOccpuancy(matcher_.px_cur_);
    }

    // 如果当前Seed对应edgelet特征，更新其grad信息
    if (it->ftr->type == Feature::EDGELET) {
      it->ftr->grad_cur_ = (matcher_.A_cur_ref_ * it->ftr->grad).normalized();
    }

    // if the seed has converged, we initialize a new candidate point and remove
    // the seed
    // 当Seed的inverse depth variance小于某个阈值时即认为收敛
    if (sqrt(it->sigma2) <
        it->z_range / options_.seed_convergence_sigma2_thresh) {
      assert(it->ftr->point == NULL);  // TODO this should not happen anymore
      Vector3d xyz_world(it->ftr->frame->T_f_w_.inverse() *
                         (it->ftr->f * (1.0 / it->mu)));

      if (it->ftr->type == Feature::EDGELET) {
        it->ftr->grad = it->ftr->grad_cur_;  //  edgelete in newkeyframe, it's
                                             //  direction should change;
      }

      Point* point = new Point(xyz_world, it->ftr);
      it->ftr->point = point;
      /* FIXME it is not threadsafe to add a feature to the frame here.
      if(frame->isKeyframe())
      {
        Feature* ftr = new Feature(frame.get(), matcher_.px_cur_,
      matcher_.search_level_);
        ftr->point = point;
        point->addFrameRef(ftr);
        frame->addFeature(ftr);
        it->ftr->frame->addFeature(it->ftr);
      }
      else
      */
      {
        // MapPointCandidates::newCandidatePoint
        seed_converged_cb_(point, it->sigma2);  // put in candidate list
      }
      it = seeds_.erase(it);
    } else if (isnan(z_inv_min)) {
      SVO_WARN_STREAM("z_min is NaN");
      it = seeds_.erase(it);
    } else
      ++it;
  }

  if (erase_seed > 0) {
    std::cout << "erase seed : " << erase_seed << std::endl;
  }
}

void DepthFilter::clearFrameQueue() {
  while (!frame_queue_.empty()) frame_queue_.pop();
}

void DepthFilter::getSeedsCopy(const FramePtr& frame, std::list<Seed>& seeds) {
  lock_t lock(seeds_mut_);
  for (std::list<Seed>::iterator it = seeds_.begin(); it != seeds_.end();
       ++it) {
    if (it->ftr->frame == frame.get()) seeds.push_back(*it);
  }
}

void DepthFilter::getAllSeedsCopy(std::list<Seed>& seeds) {
  lock_t lock(seeds_mut_);
  for (std::list<Seed>::iterator it = seeds_.begin(); it != seeds_.end();
       ++it) {
    seeds.push_back(*it);
  }
}

void DepthFilter::updateSeed(const float x, const float tau2, Seed* seed) {
  float norm_scale = sqrt(seed->sigma2 + tau2);
  if (std::isnan(norm_scale)) return;
  boost::math::normal_distribution<float> nd(seed->mu, norm_scale);
  float s2 = 1. / (1. / seed->sigma2 + 1. / tau2);
  float m = s2 * (seed->mu / seed->sigma2 + x / tau2);
  float C1 = seed->a / (seed->a + seed->b) * boost::math::pdf(nd, x);
  float C2 = seed->b / (seed->a + seed->b) * 1. / seed->z_range;
  float normalization_constant = C1 + C2;
  C1 /= normalization_constant;
  C2 /= normalization_constant;
  float f = C1 * (seed->a + 1.) / (seed->a + seed->b + 1.) +
            C2 * seed->a / (seed->a + seed->b + 1.);
  float e = C1 * (seed->a + 1.) * (seed->a + 2.) /
                ((seed->a + seed->b + 1.) * (seed->a + seed->b + 2.)) +
            C2 * seed->a * (seed->a + 1.0f) /
                ((seed->a + seed->b + 1.0f) * (seed->a + seed->b + 2.0f));

  // update parameters
  float mu_new = C1 * m + C2 * seed->mu;
  seed->sigma2 = C1 * (s2 + m * m) + C2 * (seed->sigma2 + seed->mu * seed->mu) -
                 mu_new * mu_new;
  seed->mu = mu_new;
  seed->a = (e - f) / (f - e / f);
  seed->b = seed->a * (1.0f - f) / f;
}

double DepthFilter::computeTau(const SE3& T_ref_cur, const Vector3d& f,
                               const double z, const double px_error_angle) {
  Vector3d t(T_ref_cur.translation());
  // 在ref坐标系下：当前帧原点和3d点之间的向量
  Vector3d a = f * z - t;
  double t_norm = t.norm();
  double a_norm = a.norm();
  // f和t的夹角
  double alpha = acos(f.dot(t) / t_norm);  // dot product
  // a和t的夹角
  double beta = acos(a.dot(-t) / (t_norm * a_norm));  // dot product
  // 考虑了噪声算出的最大beta
  double beta_plus = beta + px_error_angle;
  // 3d点在ref和cur之间的视差角(考虑了噪声,算出的最小gamma)
  double gamma_plus = PI - alpha - beta_plus;  // triangle angles sum to PI
  // 考虑了噪声能够算出的最大的z
  double z_plus = t_norm * sin(beta_plus) / sin(gamma_plus);  // law of sines
  // Question: tau的物理含义是什么
  // 一个pixel的观测误差最多会产生多大的深度估计误差
  return (z_plus - z);  // tau
}

}  // namespace svo
