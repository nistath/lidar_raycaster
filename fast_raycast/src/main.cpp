#include <assert.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/Eigen>
#include <Eigen/Geometry>
#include <experimental/filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <thread>

using namespace std::chrono_literals;

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <pcl/visualization/cloud_viewer.h>
#pragma GCC diagnostic pop

namespace lcaster {

using namespace Eigen;

using el_t = float;
using StdVector = std::vector<el_t, Eigen::aligned_allocator<el_t>>;
using Vector3e = Matrix<el_t, 3, 1>;
using RowVector3e = Matrix<el_t, 1, 3>;
using MatrixXe = Matrix<el_t, Dynamic, Dynamic>;
using Pair = std::pair<el_t, el_t>;

// constexpr auto toMap(StdVector& vec) {
//   return Map<Array<el_t, Dynamic, 1>, AlignmentType::Aligned>(vec.data(),
//                                                               vec.size(), 1);
// }

constexpr el_t nan(const char* tagp = "") {
  if constexpr (std::is_same_v<float, el_t>) {
    return std::nanf(tagp);
  }

  if constexpr (std::is_same_v<double, el_t>) {
    return std::nan(tagp);
  }

  if constexpr (std::is_same_v<long double, el_t>) {
    return std::nanl(tagp);
  }

  static_assert(std::is_same_v<float, el_t> || std::is_same_v<double, el_t> ||
                    std::is_same_v<long double, el_t>,
                "Invalid el_t!");
}

el_t gaussian(el_t x, el_t mean, el_t var = 1) {
  el_t toReturn = 1 / (sqrt(var * 2 * M_PI));
  el_t temp = (x - mean) / sqrt(var);
  temp *= temp;
  temp = -temp / 2;
  return toReturn * exp(temp);
}

template <int NRays = Dynamic>
class Rays : public Matrix<el_t, NRays, 6> {
 private:
  using Base = Matrix<el_t, NRays, 6>;

#define __RAYS__GET_BLOCK                                  \
  if constexpr (NRays == Dynamic) {                        \
    return this->block(0, col_offrays, Base::rows(), 3);   \
  } else {                                                 \
    return ((Base*)this)->block<NRays, 3>(0, col_offrays); \
  }

  auto constexpr get_block(size_t col_offrays) { __RAYS__GET_BLOCK }
  auto const constexpr get_block(size_t col_offrays) const { __RAYS__GET_BLOCK }

#undef __RAYS__GET_BLOCK

 public:
  RowVector3e origin_offset = {0, 0, 0};

  Rays() : Base(1, 6) {}

  template <typename OtherDerived>
  Rays(const Eigen::MatrixBase<OtherDerived>& other) : Base(other) {}

  Rays(Index rows) : Base(rows, 6) {}

  template <typename OtherDerived>
  Base& operator=(const Eigen::MatrixBase<OtherDerived>& other) {
    this->Base::operator=(other);
    return *this;
  }

  /**
   *! Returns a block of the coordinates of each ray's origin.
   */
  auto constexpr originsRaw() { return get_block(0); }
  auto const constexpr originsRaw() const { return get_block(0); }

  auto constexpr origins() { return originsRaw().rowwise() + origin_offset; }
  auto const constexpr origins() const {
    return originsRaw().rowwise() + origin_offset;
  }

  /**
   *! Returns a block of the (unit) vector of each ray's direction.
   */
  auto constexpr directions() { return get_block(3); }
  auto const constexpr directions() const { return get_block(3); }

  auto constexpr rays() const { return Base::rows(); }
};

namespace Intersection {

/**
 * Solutions
 *! Provides the scalar t for each ray such that direction * t + origin
 *! is the intersection point.
 */
template <int NRays = Dynamic, typename T = el_t>
using Solutions = Array<T, NRays, 1>;

template <int NRays = Dynamic, typename T = el_t>
constexpr Solutions<NRays, T> make_solutions(Rays<NRays> const& rays) {
  return {rays.rays(), 1};
}

/**
 * Points
 *! Provides the point each ray intersects with as computed by some method.
 */
template <int NRays = Dynamic>
using Points = Array<el_t, NRays, 3>;

constexpr int kHistogramBins = 20;
// template <int NBins = kHistogramBins>
using Histogram = std::vector<el_t>;

void printHistograms(std::vector<Histogram> const& histograms) {
  for (int c = 0; c < histograms.size(); ++c) {
    std::cout << "Threshold : "
              << *std::max_element(histograms[c].begin(), histograms[c].end())
              << endl;

    for (int l = 0; l < histograms[c].size(); ++l) {
      std::cout << "Number of points: " << histograms[c][l] << " ";
      for (int i = 0; i < histograms[c][l]; ++i) {
        std::cout << "*";
      }
      std::cout << endl;
    }
  }
}

void normalize(Histogram& h) {
  int size = std::accumulate(h.begin(), h.end(), 0);

  for (int i = 0; i < kHistogramBins; ++i) {
    h[i] /= size;
  }
}

using PointCloud = pcl::PointCloud<pcl::PointXYZ>;

template <int NRays = Dynamic>
auto computePoints(Rays<NRays> const& rays, Solutions<NRays> const& solutions) {
  return rays.origins() +
         (rays.directions().array().colwise() * solutions).matrix();
}

template <int NRays = Dynamic>
void computePoints(Rays<NRays> const& rays,
                   Solutions<NRays> const& solutions,
                   Points<NRays>& points) {
  points = computePoints(rays, solutions);
}

template <int NRays = Dynamic>
void computePoints(Rays<NRays> const& rays,
                   Solutions<NRays> const& solutions,
                   PointCloud& cloud) {
  cloud.resize(rays.rays());
  cloud.getMatrixXfMap().block(0, 0, 3, rays.rays()) =
      computePoints(rays, solutions).transpose();
}

namespace Obstacle {

/**
 * Plane
 *! A plane obstacle defined by a unit normal vector and an origin.
 */
class Plane {
 public:
  Matrix<el_t, 3, 1> const normal_;
  Matrix<el_t, 1, 3> origin_;

  Plane(Vector3e normal, Vector3e origin) : normal_{normal}, origin_{origin} {
    assert(normal_.norm() == 1);
  }

  template <int NRays = Dynamic>
  void computeSolution(Rays<NRays> const& rays,
                       Solutions<NRays>& solutions) const {
    solutions =
        (((-rays.origins()).rowwise() + origin_).matrix() * normal_).array() /
        (rays.directions().matrix() * normal_).array();
    solutions = (solutions < 0).select(nan(), solutions);
  }
};

/**
 * Cone
 *
 * Notes for implementation are found at
 * https://www.geometrictools.com/Documentation/IntersectionLineCone.pdf
 */
class Cone {
 public:
  Matrix<el_t, 3, 1> vertex_;
  Matrix<el_t, 3, 1> const direction_;
  el_t const height_;
  el_t const base_radius_;

  Cone(Vector3e vertex, Vector3e direction, el_t height, el_t baseRadius)
      : vertex_{vertex},
        direction_{direction},
        height_{height},
        base_radius_{baseRadius},
        M_{direction_ * direction_.transpose() -
           (height_ / std::hypotf(height_, base_radius_)) *
               Matrix<el_t, 3, 3>::Identity()} {}

  template <int NRays = Dynamic>
  void computeSolution(
      Rays<NRays> const& rays,
      Intersection::Solutions<NRays>& solutions,
      bool height_limit = true,
      Intersection::Solutions<NRays>* hit_height_ptr = nullptr) const {
    // Below matrices are shape (3, NRays)
    auto P = rays.origins().transpose();
    auto U = rays.directions().transpose();
    Matrix<el_t, 3, NRays> L = P.colwise() - vertex_;  // Δ from notes

    using Coeffs = Intersection::Solutions<NRays>;

    Coeffs c2 = (U.transpose() * M_ * U).diagonal();
    Coeffs c1 = (U.transpose() * M_ * L).diagonal();
    Coeffs c0 = (L.transpose() * M_ * L).diagonal();

    auto dis = (c1 * c1 - c0 * c2).sqrt();
    Intersection::Solutions<NRays> soln1 = ((-c1 - dis) / c2);
    soln1 = (soln1 < 0).select(nan(), soln1);
    Intersection::Solutions<NRays> soln2 = ((-c1 + dis) / c2);
    soln2 = (soln2 < 0).select(nan(), soln2);
    solutions = (soln1 < soln2).select(soln1, soln2);

    {
      Intersection::Solutions<NRays> hit_height_;

      if (hit_height_ptr == nullptr) {
        hit_height_ptr = &hit_height_;
      }

      Intersection::Solutions<NRays>& hit_height = *hit_height_ptr;

      hit_height = (L.transpose() * direction_).array() +
                   (solutions * (U.transpose() * direction_).array());

      if (height_limit) {
        solutions = ((0 <= hit_height) && (hit_height <= height_))
                        .select(solutions, nan());
      }
    }
  }

  Histogram createOptimalHistogram() const {
    Histogram h(kHistogramBins);
    for (int i = 0; i < kHistogramBins; ++i) {
      h[i] = 2 * base_radius_ * i / kHistogramBins;
    }

    normalize(h);
    return h;
  }

 private:
  Matrix<el_t, 3, 3> const M_;
};

}  // namespace Obstacle

namespace World {

template <int NRays = Dynamic>

class Lidar {
 public:
  // angular resolution in radians
  el_t angular_resolution_;
  // elevation angles in radians
  std::vector<el_t> elev_angles;
  Rays<Dynamic> rays_;
  // Horizontal field of view (180 degrees to simulate car occlusion)
  el_t HFOV = M_PI;
  Quaternion<el_t> direction_;

  Lidar() : Lidar{{0, 0, 0}, Quaternion<el_t>::Identity(), 1} {}

  Lidar(Vector3e origin, Quaternion<el_t> direction, el_t angular_resolution){
    rays_.origin_offset = origin;
    angular_resolution_ = angular_resolution;
    direction_ = direction;
  }

  Lidar(Vector3e origin, Quaternion<el_t> direction, el_t angular_resolution, std::string calib_info):
    Lidar(origin, direction, angular_resolution){
    setElevAngles(calib_info);
    rays_ = makeRays();
  }

  Rays<Dynamic>& rays() { return rays_; }

  Vector3e origin() { return rays_.origin_offset; }

  void setOrigin(Vector3e new_origin) { rays_.origin_offset = new_origin; }

  Quaternion<el_t> direction() { return direction_; }

  void setDirection(Quaternion<el_t> new_direction) { direction_ = new_direction; }

  int numLasers() { return elev_angles.size(); }

  void setElevAngles(std::string calib_info){
    /*
      Read file, parse elevation angles
      Assumes data in CSV format, with 1st column being laser
      number and second column being elevation angle
    */
    // TODO check if legal file name?
    std::ifstream file(calib_info);
    std::string headers;
    getline(file, headers);
    std::string container;
    std::stringstream intermediate;
    std::vector<std::string> split_info;
    while (getline(file, container)) {
      intermediate << container;
      while (getline(intermediate, container, ',')) {
        split_info.push_back(container);
      }
      intermediate.clear();
      elev_angles.push_back(stof(split_info[1]) * M_PI / 180.0);
      split_info.clear();
    }
  }

  Rays<Dynamic> makeRays() {
    /*
      Create Rays object from LiDAR angular resolution and elevation angles
    */
    // Create unit vectors
    Rays<Dynamic> rays;
    int horiz_lasers = ceil(HFOV / angular_resolution_);
    int num_rays = numLasers() * horiz_lasers;
    rays = Rays(num_rays);
    rays.origin_offset = rays_.origin_offset;
    rays.originsRaw() = MatrixXe::Zero(num_rays, 3);

    for (Index laser = 0; laser < elev_angles.size(); laser++) {
      const el_t sin_elev = sin(elev_angles[laser]);
      const el_t cos_elev = cos(elev_angles[laser]);
      for (Index i = 0; i < horiz_lasers; i++) {
        el_t phase = i * angular_resolution_;
        rays.directions()(laser * horiz_lasers + i, 0) =
            cos_elev * cos(phase);
        rays.directions()(laser * horiz_lasers + i, 1) =
            cos_elev * sin(phase);
        rays.directions()(laser * horiz_lasers + i, 2) = sin_elev;
      }
    }
    Matrix<el_t, 3, 3> rotm = direction_.toRotationMatrix();
    rays.directions().transpose() = rotm * rays.directions().transpose();
    rays.directions().rowwise().normalize();
    return rays;
  }
};

template <int NRays = Dynamic>
auto createHistogram(Obstacle::Cone const& cone,
                     std::vector<Index> const& rays_on_cone,
                     Solutions<NRays> const& hit_height) {
  Histogram histogram(kHistogramBins, 0);

  for (auto ray_idx : rays_on_cone) {
    for (int i = 0; i < kHistogramBins; ++i) {
      histogram[i] +=
          gaussian(i * cone.height_ / kHistogramBins, hit_height[ray_idx]);
    }
  }

  return histogram;
}

/*
 * Computes the KL Divergence of Q from P.
 * https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Definition
 */
el_t klDivergence(Histogram const& p, Histogram const& q) {
  int sum_p = std::accumulate(p.begin(), p.end(), 0);
  int sum_q = std::accumulate(q.begin(), q.end(), 0);

  el_t sum = 0;

  for (Index i = 0; i < kHistogramBins; ++i) {
    el_t const P = p[i] / sum_p;
    el_t const Q = q[i] / sum_q;

    sum += P * std::log(P / std::max((el_t)0.01, Q));
  }

  return sum;
}

template <int NRays = Dynamic, typename idx_t = size_t>
using ObjectIdxs = Solutions<NRays, idx_t>;

class DV {
 public:
  Obstacle::Plane plane_;
  std::vector<Obstacle::Cone> const cones_;
  std::vector<Histogram> const optimal_histograms_;

  DV(Obstacle::Plane plane, std::initializer_list<Obstacle::Cone> cones)
      : plane_{plane},
        cones_{cones},
        optimal_histograms_{createOptimalHistograms()} {}

  DV(Obstacle::Plane plane, std::vector<Obstacle::Cone> const& cones)
      : plane_{plane},
        cones_{cones},
        optimal_histograms_{createOptimalHistograms()} {}

  DV() : DV{{{0, 0, 1}, {0, 0, 0}}, {}} {}

  static DV readConeFromFile(std::string const fileName) {
    Obstacle::Plane ground({0, 0, 1}, {0, 0, 0});
    std::string x, y, z;
    ifstream in_file;
    std::vector<Obstacle::Cone> cone_list;
    in_file.open(fileName);
    if (!in_file) {
      std::cout << "Unable to open file";
      exit(1);
    }
    while (in_file.good()) {
      std::getline(in_file, x, ',');
      std::getline(in_file, y, ',');
      std::getline(in_file, z);

      cone_list.push_back(Obstacle::Cone(
          lcaster::Vector3e(std::stof(x), std::stof(y), std::stof(z)),
          lcaster::Vector3e(0, 0, -1), std::stof(z), (el_t)0.08));
    }
    in_file.close();
    return DV(ground, {cone_list});
  }

  template <int NRays = Dynamic>
  void computeSolution(Rays<NRays> const& rays,
                       Solutions<NRays>& solutions,
                       Solutions<NRays>& hit_height,
                       ObjectIdxs<NRays>& object) const {
    Solutions<NRays> solutions_temp = make_solutions(rays);
    Solutions<NRays> hit_height_temp = make_solutions(rays);

    solutions.resize(rays.rays(), 1);
    hit_height.resize(rays.rays(), 1);

    using ObjectIdxs_T = std::remove_reference_t<decltype(object)>;
    using idx_t = typename ObjectIdxs_T::Scalar;
    object = ObjectIdxs_T::Constant(rays.rays(), 1,
                                    std::numeric_limits<idx_t>::max());
    plane_.computeSolution(rays, solutions);

    for (size_t c = 0; c < cones_.size(); ++c) {
      auto const& cone = cones_[c];
      cone.computeSolution(rays, solutions_temp, true, &hit_height_temp);

      for (Index i = 0; i < rays.rays(); ++i) {
        if (std::isnan(solutions[i]) || std::isnan(solutions_temp[i]) ||
            solutions_temp[i] > solutions[i]) {
          continue;
        }

        solutions[i] = solutions_temp[i];
        hit_height[i] = hit_height_temp[i];
        object[i] = c;
      }
    }
  }

  template <int NRays = Dynamic>
  void computeRayPerCone(ObjectIdxs<NRays>& object,
                         std::vector<std::vector<Index>>& ray_per_cone) {
    ray_per_cone.resize(cones_.size());
    for (int i = 0; i < object.size(); ++i) {
      if (object[i] >= cones_.size())
        continue;
      ray_per_cone[object[i]].push_back(i);
    }
  }

  template <int NRays = Dynamic>
  auto createHistograms(std::vector<std::vector<Index>> const& ray_per_cone,
                        Solutions<NRays> const& hit_height) {
    std::vector<Histogram> histograms(cones_.size());

    for (Index c = 0; c < cones_.size(); ++c) {
      histograms[c] = createHistogram(cones_[c], ray_per_cone[c], hit_height);
    }

    return histograms;
  }

 private:
  std::vector<Histogram> createOptimalHistograms() {
    std::vector<Histogram> opt_hist(cones_.size());
    for (Index c = 0; c < cones_.size(); ++c) {
      opt_hist[c] = cones_[c].createOptimalHistogram();
    }
    return opt_hist;
  }
};  // namespace World

struct Range {
  el_t low;
  el_t high;
  el_t step;

  Index count() const { return (high - low) / step; }
  el_t get(Index iter) { return iter * step + low; }
};

class Grid {
 public:
  std::array<Range, 2> ranges_;
  MatrixXd grid_;

  Grid(std::array<Range, 2> ranges)
      : ranges_{ranges}, grid_{ranges[0].count(), ranges[1].count()} {}

  void serialize(std::ostream& stream) {
    for (auto const& range : ranges_) {
      stream << range.low << "," << range.high << "," << range.step << "\n";
    }

    const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");
    stream << grid_.format(CSVFormat);
  }

  void serialize(std::string const& out) {
    std::ofstream file(out);
    serialize(file);
  }
};

template <int NRays = Dynamic>
auto computeGrid(DV const& dv, Rays<NRays>& rays, std::array<Range, 2> ranges) {
  Grid grid(ranges);

  Solutions<NRays> solutions;
  Solutions<NRays> hit_height;
  ObjectIdxs<NRays> object;

  std::atomic<Index> iters_done = 0;

  auto const origin_offset = rays.origin_offset;

#pragma omp parallel for
  for (Index y_iter = 0; y_iter < ranges[0].count(); ++y_iter) {
    for (Index z_iter = 0; z_iter < ranges[1].count(); ++z_iter) {
      rays.origin_offset = origin_offset + RowVector3e{0, ranges[0].get(y_iter),
                                                       ranges[1].get(z_iter)};

      dv.computeSolution(rays, solutions, hit_height, object);
      std::vector<std::vector<Index>> ray_per_cone;
      dv.computeRayPerCone(object, ray_per_cone);

      grid.grid_(y_iter, z_iter) = std::accumulate(
          ray_per_cone.begin(), ray_per_cone.end(), (Index)0,
          [](Index prior, std::vector<Index> const& vec) -> Index {
            return prior + vec.size();
          });
    }

    std::cout << 100 * (++iters_done) / ranges[0].count() << "%\n";
  }

  return grid;
}

}  // namespace World

}  // namespace Intersection
}  // namespace lcaster

int main() {
  using namespace lcaster;
  using namespace lcaster::Intersection;

  World::Lidar vlp32({0, 0, 0.3}, {1,0,0,0}, 0.2 * M_PI / 180.0, "../sensor_info/VLP32_LaserInfo.csv");
  Rays<Dynamic>& rays = vlp32.rays();

  el_t height = 0;
  std::cin >> height;
  vlp32.setOrigin({0, 0, height});
  // rays.origin_offset = {0, 0, height};

  Solutions<Dynamic> solutions(rays.rays());
  Solutions<Dynamic> hit_height(rays.rays());
  World::ObjectIdxs<Dynamic> object;

  World::DV world = World::DV::readConeFromFile("../src/test.csv");
  world.computeSolution(rays, solutions, hit_height, object);
  std::vector<std::vector<Index>> ray_per_cone;
  world.computeRayPerCone(object, ray_per_cone);

  auto p = world.createHistograms(ray_per_cone, hit_height);
  printHistograms(p);

  PointCloud::Ptr cloud(new PointCloud);
  computePoints(rays, solutions, *cloud);

  pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
  viewer.showCloud(cloud);
  while (!viewer.wasStopped()) {
    std::this_thread::sleep_for(100ms);
  }

  return 0;
}
