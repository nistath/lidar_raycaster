#include <assert.h>
#include <eigen3/Eigen/Eigen>
#include <iostream>

using namespace Eigen;

namespace lcaster {

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
  Matrix<float, 3, 1> plane_normal;
  Matrix<float, 1, 3> plane_origin;

  Plane(Vector3f const& normal, Vector3f const& origin)
      : plane_normal{normal}, plane_origin{origin} {
    assert(plane_normal.norm() == 1);
  }

  template <int NRays = Dynamic>
  void computeSolution(Rays<NRays> const& rays,
                       Intersection::Solutions<NRays>& solutions) {
    solutions = (((-rays.origins()).rowwise() + plane_origin) * plane_normal) /
                (rays.directions() * plane_normal)(0);
  }
};

}  // namespace Obstacle
}  // namespace Intersection
}  // namespace lcaster

int main() {
  using namespace lcaster;
  using namespace lcaster::Intersection;

  Rays<Dynamic> rays = Rays<10>::Zero();
  rays.origins().col(2) = decltype(rays.origins().col(2))::Ones(10, 1);

  for (int i = 0; i < rays.rays(); ++i) {
    auto dir = 2 * M_PI * i / rays.rays();
    rays.directions()(i, 0) = sin(dir);
    rays.directions()(i, 1) = cos(dir);
    rays.directions()(i, 2) = -1;
  }

  std::cout << rays << "\n";

  Obstacle::Plane ground({0, 0, 1}, {0, 0, 0});

  Solutions<Dynamic> solutions(rays.rays());
  ground.computeSolution(rays, solutions);
  std::cout << solutions << "\n";

  Points<Dynamic> points(rays.rays(), 3);
  computePoints(rays, solutions, points);
  std::cout << points << "\n";

  return 0;
}
