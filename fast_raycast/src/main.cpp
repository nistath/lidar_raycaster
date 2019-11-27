#include <assert.h>
#include <chrono>
#include <eigen3/Eigen/Eigen>
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
 *! Provides the scalar `t` for each ray such that direction * t + origin
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

 private:
  Matrix<el_t, 3, 3> const M_;
};

}  // namespace Obstacle

namespace World {

template <int NRays = Dynamic>
using ObjectIdxs = Solutions<NRays, Index>;

class Lidar {
 public:
  Vector3e origin_;
  int num_lasers_;
  // angular resolution in radians
  el_t angular_resolution_;
  // elevation angles in radians
  std::vector<el_t> elev_angles;
  Rays<Dynamic> rays_;

  Lidar() : Lidar{{0, 0, 0}, 1, 1} {}

  Lidar(Vector3e origin, int num_lasers, el_t angular_resolution)
      : origin_{origin},
        num_lasers_{num_lasers},
        angular_resolution_{angular_resolution} {}

  Rays<Dynamic>& rays() { return this->rays_; }

  Vector3e origin() { return this->origin_; }

  void setRays(std::string csv) {
    /*
      Create Rays object from LiDAR calibration info
      Assumes data in CSV format, with 1st column being laser
      number and second column being elevation angle
    */
    // Read file, parse elevation angles
    std::ifstream file(csv);
    std::string headers;
    getline(file, headers);
    std::string laser_info[this->num_lasers_ - 1];
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

    // Create unit vectors
    int horiz_lasers = ceil((2 * M_PI) / this->angular_resolution_);
    int num_rays = this->num_lasers_ * horiz_lasers;
    this->rays_ = Rays(num_rays);
    this->rays_.originsRaw() = this->origin_.transpose().replicate(num_rays, 1);

    for (Index laser = 0; laser < elev_angles.size(); laser++) {
      const el_t sin_elev = sin(elev_angles[laser]);
      const el_t cos_elev = cos(elev_angles[laser]);
      for (Index i = 0; i < horiz_lasers; i++) {
        el_t phase = i * (this->angular_resolution_);
        this->rays_.directions()(laser * horiz_lasers + i, 0) =
            cos_elev * cos(phase);
        this->rays_.directions()(laser * horiz_lasers + i, 1) =
            cos_elev * sin(phase);
        this->rays_.directions()(laser * horiz_lasers + i, 2) = sin_elev;
      }
    }
    this->rays_.directions().rowwise().normalize();
  }
};

class DV {
 public:
  Obstacle::Plane plane_;
  std::vector<Obstacle::Cone> cones_;

  DV(Obstacle::Plane plane, std::initializer_list<Obstacle::Cone> cones)
      : plane_{plane}, cones_{cones} {}

  DV() : DV{{{0, 0, 1}, {0, 0, 0}}, {}} {}

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
  auto computeIdxsPerCone(ObjectIdxs<NRays> const& object) const {
    std::vector<std::vector<Index>> idxs_per_cone(cones_.size());

    for (Index i = 0; i < object.size(); ++i) {
      if (object[i] >= cones_.size())
        continue;

      idxs_per_cone[object[i]].push_back(i);
    }

    return idxs_per_cone;
  }
};

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
      auto idxs_per_cone = dv.computeIdxsPerCone(object);

      grid.grid_(y_iter, z_iter) = std::accumulate(
          idxs_per_cone.begin(), idxs_per_cone.end(), (Index)0,
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

  World::Lidar vlp32({0, 0, 0}, 32, 0.2 * M_PI / 180.0);
  vlp32.setRays("fast_raycast/sensor_info/VLP32_LaserInfo.csv");
  Rays<Dynamic>& rays = vlp32.rays();

  Obstacle::Plane ground({0, 0, 1}, {0, 0, 0});

  constexpr el_t sc = 0.75;

  bool earr;
  std::cin >> earr;

  std::unique_ptr<World::DV> world_ptr;
  if (earr) {
    world_ptr.reset(new World::DV(
        ground,
        {
            {{-12.75835829 * sc, 18.81545688 * sc, 0.29},
             {0, 0, -1},
             0.29,
             0.08},
            {{-8.27767283 * sc, 17.59812307 * sc, 0.29},
             {0, 0, -1},
             0.29,
             0.08},
            {{-4.26349242 * sc, 16.15671487 * sc, 0.29},
             {0, 0, -1},
             0.29,
             0.08},
            {{-0.99153611 * sc, 14.40308463 * sc, 0.29},
             {0, 0, -1},
             0.29,
             0.08},
            {{1.12454352 * sc, 12.2763364 * sc, 0.29}, {0, 0, -1}, 0.29, 0.08},
            {{2.01314521 * sc, 9.33822398 * sc, 0.29}, {0, 0, -1}, 0.29, 0.08},
            {{1.98908967 * sc, 5.84530425 * sc, 0.29}, {0, 0, -1}, 0.29, 0.08},
            {{1.43369938 * sc, 1.8860828 * sc, 0.29}, {0, 0, -1}, 0.29, 0.08},
            {{-13.47161573 * sc, 15.90147955 * sc, 0.29},
             {0, 0, -1},
             0.29,
             0.08},
            {{-9.16834254 * sc, 14.73338793 * sc, 0.29},
             {0, 0, -1},
             0.29,
             0.08},
            {{-5.45778135 * sc, 13.40468399 * sc, 0.29},
             {0, 0, -1},
             0.29,
             0.08},
            {{-2.73894015 * sc, 11.96452376 * sc, 0.29},
             {0, 0, -1},
             0.29,
             0.08},
            {{-1.46294141 * sc, 10.75813953 * sc, 0.29},
             {0, 0, -1},
             0.29,
             0.08},
            {{-0.973512 * sc, 9.05559578 * sc, 0.29}, {0, 0, -1}, 0.29, 0.08},
            {{-0.99791948 * sc, 6.12418841 * sc, 0.29}, {0, 0, -1}, 0.29, 0.08},
            {{-1.51764477 * sc, 2.42419778 * sc, 0.29}, {0, 0, -1}, 0.29, 0.08},
        }));
  } else {
    world_ptr.reset(new World::DV(
        ground, {
                    {{-5.770087622473654 * sc, 13.960476554628654 * sc, 0.29},
                     {0, 0, -1},
                     0.29,
                     0.08},
                    {{-3.5531794846979197 * sc, 10.09163739418124 * sc, 0.29},
                     {0, 0, -1},
                     0.29,
                     0.08},
                    {{-1.8689520677436653 * sc, 6.528338851853511 * sc, 0.29},
                     {0, 0, -1},
                     0.29,
                     0.08},
                    {{-0.9647880812999942 * sc, 3.5874807367224406 * sc, 0.29},
                     {0, 0, -1},
                     0.29,
                     0.08},
                    {{-0.9155631877776976 * sc, 1.8321709002122402 * sc, 0.29},
                     {0, 0, -1},
                     0.29,
                     0.08},
                    {{-3.20524532871408 * sc, 15.516620509454873 * sc, 0.29},
                     {0, 0, -1},
                     0.29,
                     0.08},
                    {{-0.8977072413626149 * sc, 11.487512447642604 * sc, 0.29},
                     {0, 0, -1},
                     0.29,
                     0.08},
                    {{0.9215174312005985 * sc, 7.629828740804735 * sc, 0.29},
                     {0, 0, -1},
                     0.29,
                     0.08},
                    {{1.9951360927293942 * sc, 4.076202420297469 * sc, 0.29},
                     {0, 0, -1},
                     0.29,
                     0.08},
                    {{1.9875922542497209 * sc, 1.0760700607559186 * sc, 0.29},
                     {0, 0, -1},
                     0.29,
                     0.08},
                }));
  }
  World::DV& world = *world_ptr.get();

  el_t xl, xh, xs;
  el_t yl, yh, ys;
  std::cin >> xl >> xh >> xs >> yl >> yh >> ys;
  std::cout << "Computing grid...\n";
  auto grid = World::computeGrid(
      world, rays, {World::Range{xl, xh, xs}, World::Range{yl, yh, ys}});

  std::string fname;
  std::cin >> fname;
  grid.serialize(fname);

  std::cout << "Computing sample cloud...\n";

  el_t height = 0;
  std::cin >> height;
  rays.origin_offset = {0, 0, height};

  Solutions<Dynamic> solutions(rays.rays());
  Solutions<Dynamic> hit_height(rays.rays());
  World::ObjectIdxs<Dynamic> object;
  world.computeSolution(rays, solutions, hit_height, object);
  // ground.computeSolution(rays, solutions);

  PointCloud::Ptr cloud(new PointCloud);
  computePoints(rays, solutions, *cloud);

  pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
  viewer.showCloud(cloud);
  while (!viewer.wasStopped()) {
    std::this_thread::sleep_for(100ms);
  }

  return 0;
}
