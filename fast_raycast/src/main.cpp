#include <assert.h>
#include <eigen3/Eigen/Eigen>
#include <iostream>

namespace lcaster {

using namespace Eigen;

template <int NRays = Dynamic>
class Rays : public Matrix<float, NRays, 6> {
 private:
  using Base = Matrix<float, NRays, 6>;

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
using Solutions = Array<float, NRays, 1>;

/**
 * Points
 *! Provides the point each ray intersects with as computed by some method.
 */
template <int NRays = Dynamic>
using Points = Matrix<float, NRays, 3>;

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
  Matrix<float, 3, 1> const normal_;
  Matrix<float, 1, 3> origin_;

  Plane(Vector3f const& normal, Vector3f const& origin)
      : normal_{normal}, origin_{origin} {
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
 */
// template <bool BothSides = false, bool LimitHeight = false>
class Cone {
 public:
  Matrix<float, 3, 1> const vertex_;
  Matrix<float, 3, 1> const direction_;
  float const height_;
  float const base_radius_;

  Cone(Vector3f vertex, Vector3f direction, float height, float baseRadius)
      : vertex_{vertex},
        direction_{direction},
        height_{height},
        base_radius_{baseRadius},
        M_{direction_ * direction_.transpose() -
           (height_ / std::hypotf(height_, base_radius_)) *
               Matrix<float, 3, 3>::Identity()} {}

  template <int NRays = Dynamic>
  void computeSolution(Rays<NRays> const& rays,
                       Intersection::Solutions<NRays>& solutions) {
    // P is 3 by NRays
    #define P (rays.origins().transpose())
    // U is 3 by NRays
    #define U (rays.directions().transpose())
    // DELTA is 3 by NRays
    Matrix<float, 3, NRays> delta = P.colwise() - vertex_;

    using Coeffs = Intersection::Solutions<NRays>;

    Coeffs c2 = (U.transpose() * M_ * U).diagonal();
    Coeffs c1 = (U.transpose() * M_ * delta).diagonal();
    Coeffs c0 = (delta.transpose() * M_ * delta).diagonal();

    solutions = (-c1 - (c1 * c1 - c0 * c2).sqrt()) / c2;
    #undef P
    #undef U
  }

 private:
  Matrix<float, 3, 3> const M_;
};

}  // namespace Obstacle
}  // namespace Intersection
}  // namespace lcaster

#include <chrono>

int main() {
  using namespace lcaster;
  using namespace lcaster::Intersection;

  constexpr size_t NRays = 10000000;
  Rays<Dynamic> rays = Rays<NRays>::Zero();
  rays.origins().col(2) = decltype(rays.origins().col(2))::Ones(NRays, 1);

  for (int i = 0; i < rays.rays(); ++i) {
    auto dir = 2 * M_PI * i / rays.rays();
    rays.directions()(i, 0) = sin(dir);
    rays.directions()(i, 1) = cos(dir);
    rays.directions()(i, 2) = -1;
  }

  // std::cout << rays << "\n";

  Obstacle::Plane ground({0, 0, 1}, {0, 0, 0});
  Obstacle::Cone cone({0, 0, 1}, {0, 0, -1}, 0.35, 0.08);

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

  std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end1 -
                                                                    start1)
                   .count()
            << ",";
  std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end2 -
                                                                    start2)
                   .count()
            << "\n";

  return 0;
}
