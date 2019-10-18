#include <eigen3/Eigen/Eigen>
#include <iostream>

using namespace Eigen;

template <int NRays = Dynamic>
class RaySet : public Matrix<float, NRays, 6> {
 private:
  using Base = Matrix<float, NRays, 6>;

  auto constexpr get_block(size_t col_offset) {
    if constexpr (NRays == Dynamic) {
      return this->block(0, col_offset, this->rows(), 3);
    } else {
      return this->block<NRays, 3>(0, col_offset);
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
};

template <int NRays = Dynamic>
using IntersectionSolution = Matrix<float, NRays, 1>;

class PlaneIntersector {
 public:
  Vector3f plane_normal;
  Vector3f plane_origin;

  PlaneIntersector(Vector3f const& origin, Vector3f const& normal)
      : plane_normal{normal}, plane_origin{origin} {}

  template <int NRays = Dynamic>
  void computeSolution(RaySet<NRays> const& rayset,
                       IntersectionSolution<NRays>& solution) {
    // solution = plane_origin - rayset;
  }
};

int main() {
  RaySet<Dynamic> set(4);
  set << 1,2,3,4,5,6,
         1,2,3,4,5,6,
         1,2,3,4,5,6,
         1,2,3,4,5,6;
  std::cout << set.origins() << "\n";
  std::cout << set.directions() << "\n";

  return 0;
}
