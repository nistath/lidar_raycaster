#include <assert.h>
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <limits>
#include <optional>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <pcl/visualization/cloud_viewer.h>
#pragma GCC diagnostic pop

namespace lcaster {

using namespace Eigen;

using el_t = float;
using Vector3e = Matrix<el_t, 3, 1>;

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
  Rays(): Base(1,6) {}

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
  auto constexpr origins() { return get_block(0); }
  auto const constexpr origins() const { return get_block(0); }

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
    solutions = ((-c1 - dis) / c2).min((-c1 + dis) / c2);

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

template <int NRays = Dynamic, typename idx_t = size_t>
using ObjectIdxs = Solutions<NRays, idx_t>;

class Lidar {
  public:
    Vector3e origin_;
    int num_lasers_;
    //angular resolution in degrees
    el_t angular_resolution_;
    std::vector<el_t> elev_angles;
    Rays<Dynamic> rays_;

    Lidar() : Lidar{{0,0,0}, 1, 1} {}

    Lidar(Vector3e origin, int num_lasers, el_t angular_resolution)
          : origin_{origin}, num_lasers_{num_lasers},
            angular_resolution_{angular_resolution} {}

    Rays<Dynamic> rays() { return this->rays_; }

    Vector3e origin() { return this->origin_; }

    void setRays(std::string csv){
      /*
        Create Rays object from LiDAR calibration info
        Assumes data in CSV format, with 1st column being laser
        number and second column being elevation angle
      */
      // Read file, parse elevation angles
      std::ifstream file (csv);
      std::string headers;
      getline(file, headers);
      std::string laser_info[this->num_lasers_ - 1];
      std::string container;
      std::stringstream intermediate;
      std::vector<std::string> split_info;
      while (getline(file, container)){
        intermediate << container;
        while (getline(intermediate, container, ',')){
          split_info.push_back(container);
        }
        intermediate.clear();
        elev_angles.push_back(stof(split_info[1]));
        split_info.clear();
      }

      //Create unit vectors
      int horizontal_lasers = 360/this->angular_resolution_;
      int num_rays = this->num_lasers_ * horizontal_lasers;
      this->rays_ = Rays(num_rays);
      this->rays_.origins() = this->origin_.transpose().replicate(num_rays, 1);

      for (int laser = 0; laser < elev_angles.size(); laser ++){
        const el_t z = sin(elev_angles[laser]);
        for (int i = 0; i < horizontal_lasers; i++){
          el_t phase = i*this->angular_resolution_;
          this->rays_.directions()(laser * horizontal_lasers + i, 0) = cos(phase);
          this->rays_.directions()(laser * horizontal_lasers + i, 1) = sin(phase);
          this->rays_.directions()(laser * horizontal_lasers + i, 2) = z;
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

    using ObjectIdxs_T = std::remove_reference_t<decltype(object)>;
    using idx_t = typename ObjectIdxs_T::Scalar;
    object = ObjectIdxs_T::Constant(rays.rays(), 1,
                                    std::numeric_limits<idx_t>::max());
    plane_.computeSolution(rays, solutions);

    for (size_t c = 0; c < cones_.size(); ++c) {
      auto const& cone = cones_[c];
      cone.computeSolution(rays, solutions_temp, true, &hit_height_temp);

      for (int i = 0; i < rays.rays(); ++i) {
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
};

}  // namespace World

}  // namespace Intersection
}  // namespace lcaster

#include <chrono>

int main() {
  using namespace lcaster;
  using namespace lcaster::Intersection;

  World::Lidar vlp32 = World::Lidar(Vector3e(0,1,0), 32, 0.2);
  vlp32.setRays("../sensor_info/VLP32_LaserInfo.csv");
  Rays<Dynamic> rays = vlp32.rays();

  // constexpr el_t HFOV = M_PI / 8;
  // constexpr el_t HBIAS = -HFOV / 2;
  // constexpr el_t VFOV = M_PI / 6;
  // constexpr el_t VBIAS = -M_PI / 2;
  //
  // constexpr int NRings = 200;
  // constexpr int NPoints = 200;
  // constexpr int NRays = NPoints * NRings;
  // Rays<Dynamic> rays = Rays<NRays>::Zero();
  // rays.origins().col(2) = decltype(rays.origins().col(2))::Ones(NRays, 1);
  //
  // for (int ring = 0; ring < NRings; ++ring) {
  //   const el_t z = -2 * cos(VFOV * ring / NRings + VBIAS) - 0.5;
  //   for (int i = 0; i < NPoints; ++i) {
  //     const el_t phase = HFOV * i / NPoints + HBIAS;
  //     rays.directions()(ring * NPoints + i, 0) = cos(phase);
  //     rays.directions()(ring * NPoints + i, 1) = sin(phase);
  //     rays.directions()(ring * NPoints + i, 2) = z;
  //   }
  // }
  //
  // rays.directions().rowwise().normalize();

  Obstacle::Plane ground({0, 0, 1}, {0, 0, 0});
  Obstacle::Cone cone({1, 0, 0.29}, {0, 0, -1}, 0.29, 0.08);

  Solutions<Dynamic> solutions(rays.rays());
  Solutions<Dynamic> hit_height(rays.rays());

  World::DV world(ground, {cone});
  World::ObjectIdxs<Dynamic> object;
  world.computeSolution(rays, solutions, hit_height, object);
  // ground.computeSolution(rays, solutions);

  PointCloud::Ptr cloud(new PointCloud);
  computePoints(rays, solutions, *cloud);

  if (false) {
    cone.computeSolution(rays, solutions);
    PointCloud::Ptr cloud2(new PointCloud);
    computePoints(rays, solutions, *cloud2);

    *cloud += *cloud2;
  }

  pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
  viewer.showCloud(cloud);
  while (!viewer.wasStopped()) {
  }

  return 0;
}
