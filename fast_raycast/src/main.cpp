#include <assert.h>
#include <eigen3/Eigen/Eigen>
#include <iostream>

using namespace Eigen;

template <int NRays = Dynamic>
class RaySet : public Matrix<float, NRays, 6> {
 private:
  using Base = Matrix<float, NRays, 6>;

  auto constexpr get_block(size_t col_offrays) {
    if constexpr (NRays == Dynamic) {
      return this->block(0, col_offrays, Base::rows(), 3);
    } else {
      return ((Base*)this)->block<NRays, 3>(0, col_offrays);
    }
  }

 public:
  template <typename OtherDerived>
  RaySet(const Eigen::MatrixBase<OtherDerived>& other) : Base(other) {}

  RaySet(Index rows) : Base(rows, 6) {}

  template <typename OtherDerived>
  Base& operator=(const Eigen::MatrixBase<OtherDerived>& other) {
    this->Base::operator=(other);
    return *this;
  }

  auto constexpr origins() { return get_block(0); }
  auto constexpr directions() { return get_block(3); }
  auto constexpr rays() { return Base::rows(); }
};

template <int NRays = Dynamic>
using IntersectionSolutions = Array<float, NRays, 1>;

template <int NRays = Dynamic>
using IntersectionPoints = Matrix<float, NRays, 3>;

class PlaneIntersector {
 public:
  Matrix<float, 3, 1> plane_normal;
  Matrix<float, 1, 3> plane_origin;

  PlaneIntersector(Vector3f const& normal, Vector3f const& origin)
      : plane_normal{normal}, plane_origin{origin} {
    assert(plane_normal.norm() == 1);
  }

  template <int NRays = Dynamic>
  void computeSolution(RaySet<NRays>/*const*/& rays,
                       IntersectionSolutions<NRays>& solutions) {
    solutions = (((-rays.origins()).rowwise() + plane_origin) * plane_normal) /
                (rays.directions() * plane_normal)(0);
  }
};

template <int NRays = Dynamic>
void computeIntersections(RaySet<NRays>/*const*/& rays,
                          IntersectionSolutions<NRays> const& solutions,
                          IntersectionPoints<NRays>& points) {
  points.noalias() = rays.origins() +
                     (rays.directions().array().colwise() * solutions).matrix();
}

int main() {
  RaySet<Dynamic> rays = RaySet<10>::Zero();
  rays.origins().col(2) = decltype(rays.origins().col(2))::Ones(10, 1);

  for (int i = 0; i < rays.rays(); ++i) {
    auto dir = 2 * M_PI * i / rays.rays();
    rays.directions()(i, 0) = sin(dir);
    rays.directions()(i, 1) = cos(dir);
    rays.directions()(i, 2) = -1;
  }

  std::cout << rays << "\n";

  PlaneIntersector ground({0, 0, 1}, {0, 0, 0});

  IntersectionSolutions<Dynamic> solutions(rays.rays());
  ground.computeSolution(rays, solutions);
  std::cout << solutions << "\n";

  IntersectionPoints<Dynamic> points(rays.rays(), 3);
  computeIntersections(rays, solutions, points);
  std::cout << points << "\n";

  return 0;
}
