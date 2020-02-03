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

#ifndef SVO_REPROJECTION_H_
#define SVO_REPROJECTION_H_

#include <svo/global.h>
#include <svo/matcher.h>

namespace svo {
class AbstractCamera;
}

namespace svo {

class Map;
class Point;

/// Project points from the map into the image and find the corresponding
/// feature (corner). We don't search a match for every point but only for one
/// point per cell. Thereby, we achieve a homogeneously distributed set of
/// matched features and at the same time we can save processing time by not
/// projecting all points.
class Reprojector {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// Reprojector config parameters
  struct Options {
    size_t max_n_kfs;  //!< max number of keyframes to reproject from
    bool find_match_direct;
    Options() : max_n_kfs(10), find_match_direct(true) {}
  } options_;

  size_t n_matches_;
  size_t n_trials_;

  Reprojector(svo::AbstractCamera* cam, Map& map);

  ~Reprojector();

  /// Project points from the map into the image. First finds keyframes with
  /// overlapping field of view and projects only those map-points.
  void reprojectMap(
      FramePtr frame,
      std::vector<std::pair<FramePtr, std::size_t> >& overlap_kfs);

 private:
  /// A candidate is a point that projects into the image plane and for which we
  /// will search a maching feature in the image.
  struct Candidate {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Point* pt;    //!< 3D point.
    Vector2d px;  //!< projected 2D pixel location.
    Candidate(Point* pt, Vector2d& px) : pt(pt), px(px) {}
  };
  // 临近map中的point投影到当前cell形成的list
  typedef std::list<Candidate, aligned_allocator<Candidate> > Cell;
  typedef std::vector<Cell*> CandidateGrid;

  /// The grid stores a set of candidate matches. For every grid cell we try to
  /// find one match.
  struct Grid {
    // 这里的Cell对应正常的grid概念,这里的Grid对应正常的grids概念
    CandidateGrid cells;
    // Question: cell_order为什么要是乱序的
    vector<int> cell_order;
    int cell_size;
    // grid_n_cols = img_cols / cell_size
    int grid_n_cols;
    // grid_n_rows = img_rows / cell_size
    int grid_n_rows;
  };

  Grid grid_;
  Matcher matcher_;
  Map& map_;

  static bool pointQualityComparator(Candidate& lhs, Candidate& rhs);
  void initializeGrid(svo::AbstractCamera* cam);
  void resetGrid();
  bool reprojectCell(Cell& cell, FramePtr frame);
  bool reprojectPoint(FramePtr frame, Point* point);
};

}  // namespace svo

#endif  // SVO_REPROJECTION_H_
