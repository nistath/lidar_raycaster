#include <assert.h>
#include <eigen3/Eigen/Eigen>
#include <iostream>

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

  auto constexpr rays() { return Base::rows(); }
};

namespace Intersection {

/**
 * Solutions
 *! Provides the scalar `t` for each ray such that direction * t + origin
 *! is the intersection point.
 */
template <int NRays = Dynamic>
using Solutions = Array<el_t, NRays, 1>;

/**
 * Points
 *! Provides the point each ray intersects with as computed by some method.
 */
template <int NRays = Dynamic>
using Points = Matrix<el_t, NRays, 3>;

template <int NRays = Dynamic>
void computePoints(Rays<NRays> const& rays,
                   Solutions<NRays> const& solutions,
                   Points<NRays>& points) {
  points.noalias() = rays.origins() +
                     (rays.directions().array().colwise() * solutions).matrix();
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
                       Intersection::Solutions<NRays>& solutions) {
    solutions = (((-rays.origins()).rowwise() + origin_) * normal_) /
                (rays.directions() * normal_)(0);
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
  void computeSolution(Rays<NRays> const& rays,
                       Intersection::Solutions<NRays>& solutions,
                       bool heightLimit = true) const {
    // Below matrices are shape (3, NRays)
    auto P = rays.origins().transpose();
    auto U = rays.directions().transpose();
    Matrix<el_t, 3, NRays> L = P.colwise() - vertex_;  // Δ from notes

    using Coeffs = Intersection::Solutions<NRays>;

    Coeffs c2 = (U.transpose() * M_ * U).diagonal();
    Coeffs c1 = (U.transpose() * M_ * L).diagonal();
    Coeffs c0 = (L.transpose() * M_ * L).diagonal();

    auto dis = (c1 * c1 - c0 * c2).sqrt();
    auto sol = ((-c1 - dis) / c2).min((-c1 + dis) / c2);

    if (heightLimit) {
      auto hit_height = (L.transpose() * direction_).array() +
                        (sol * (U.transpose() * direction_).array());

      solutions =
          ((0 <= hit_height) && (hit_height <= height_)).select(sol, nan());
    } else {
      solutions = sol;
    }
  }

 private:
  Matrix<el_t, 3, 3> const M_;
};

}  // namespace Obstacle
}  // namespace Intersection
}  // namespace lcaster

#include <chrono>

int main() {
  using namespace lcaster;
  using namespace lcaster::Intersection;

  constexpr el_t HFOV = M_PI / 8;
  constexpr el_t HBIAS = -HFOV / 2;
  constexpr el_t VFOV = M_PI / 6;
  constexpr el_t VBIAS = -M_PI / 2;

  constexpr int NRings = 20;
  constexpr int NPoints = 20;
  constexpr int NRays = NPoints * NRings;
  Rays<Dynamic> rays = Rays<NRays>::Zero();
  rays.origins().col(2) = decltype(rays.origins().col(2))::Ones(NRays, 1);

  for (int ring = 0; ring < NRings; ++ring) {
    const el_t z = -2 * cos(VFOV * ring / NRings + VBIAS) - 0.5;
    for (int i = 0; i < NPoints; ++i) {
      const el_t phase = HFOV * i / NPoints + HBIAS;
      rays.directions()(ring * NPoints + i, 0) = cos(phase);
      rays.directions()(ring * NPoints + i, 1) = sin(phase);
      rays.directions()(ring * NPoints + i, 2) = z;
    }
  }

  rays.directions().rowwise().normalize();

  Obstacle::Plane ground({0, 0, 1}, {0, 0, 0});
  Obstacle::Cone cone({1, 0, 0.29}, {0, 0, -1}, 0.29, 0.08);

  Solutions<Dynamic> solutions(rays.rays());

  auto start1 = std::chrono::high_resolution_clock::now();
  // ground.computeSolution(rays, solutions);
  cone.computeSolution(rays, solutions);
  auto end1 = std::chrono::high_resolution_clock::now();
  // std::cout << solutions << "\n";

  Points<Dynamic> points(rays.rays(), 3);
  auto start2 = std::chrono::high_resolution_clock::now();
  computePoints(rays, solutions, points);
  auto end2 = std::chrono::high_resolution_clock::now();
  // std::cout << points << "\n";

  std::cerr << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 -
                                                                    start1)
                   .count()
            << ",";
  std::cerr << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 -
                                                                    start2)
                   .count()
            << std::endl;

  // std::cout << solutions;

  return 0;
}
