#include <assert.h>
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <limits>
#include <optional>

#include <string>


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
void computePoints(Rays<NRays> const& rays, Solutions<NRays> const& solutions,
                   Points<NRays>& points) {
  points = computePoints(rays, solutions);
}

template <int NRays = Dynamic>
void computePoints(Rays<NRays> const& rays, Solutions<NRays> const& solutions,
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
                       Intersection::Solutions<NRays>& solutions) const {
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
  void computeSolution(
      Rays<NRays> const& rays, Intersection::Solutions<NRays>& solutions,
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

class DV {
 public:
  Obstacle::Plane plane_;
  std::vector<Obstacle::Cone> cones_;

  DV(Obstacle::Plane plane, std::initializer_list<Obstacle::Cone> cones)
      : plane_{plane}, cones_{cones} {}

  DV() : DV{{{0, 0, 1}, {0, 0, 0}}, {}} {}

  template <int NRays = Dynamic>
  void computeSolution(Rays<NRays> const& rays, Solutions<NRays>& solutions,
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
        if (solutions_temp[i] > solutions[i]) {
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

namespace Metrics {

using namespace Intersection;

template <int NRays>
void coneHeightMetric(Obstacle::Cone& cone, Points<NRays>& points) {
  (void)cone;
  (void)points;

// get the number of rows in the matrix points
int rownumber = points.rows();
 
// initialize a matrix to host the projected points and raw data, in which all points are zero'ed at beginning
MatrixXf ProjPointsTemp = MatrixXf::Zero(rownumber,3);
MatrixXf RawPointsTemp = MatrixXf::Zero(rownumber,3);

// setup vectors for the projected line, which is defined as AB= B-A, in this case the center line of the cone
  Vector3e A = cone.vertex_;
  Vector3e B = cone.vertex_ + cone.height_ * cone.direction_;
  Vector3e AB = B - A;
  float norm = AB.dot(AB);
  
// this is a testset instead of the large point data set to validate the codes
MatrixXf TestSet (10,3);
TestSet << 1, 1, 0,
          0.34, -0.21, 0.08,
          0.34, -0.19, 0.97,
          0.11, 3, 0.12,
          -1, 6, 0.14,
          0.55, 6.99, 0.188,
          0.33, -9.77, 0.194,
          0.33, 8.33, 0.195,
          0.8, 0.2, 0.21,
          0.23, -0.001, 0.24;
 // int rownumber = TestSet.rows();

// use for loop to project points collected on the line
int j = 0;
int s = 0;
  for (int i = 0; i < rownumber; ++i) 
  {
     Vector3e M = points.row(i).transpose();
     Vector3e AM = M - A;
     float dot = AB.dot(AM);
     float d1 = dot / norm;
     Vector3e AP = d1 * AB;
     Vector3e P = AP + A;

      // to eliminate nan from the matrix by applying if condition for the P dot product of P to be a number, at least not nan
     if ( P.dot(P) > 0.0 ) 
     {
          ProjPointsTemp.row(j) = P.transpose();
          RawPointsTemp.row(s) = points.row(i);
          j = j + 1;
          s = s + 1;
     }
  }

// define and output the size of the actual matrix with valid points in
std::cout << j << endl << endl;
int actualrowsize = j ;
std::cout << actualrowsize << endl << endl;

// use block operation to assign the valid points into newly defined ProjPoints and RawPoints matrix with the actualrow size
// keep in mind the proper usage of block operation for dynamic matrix in comparison to the fixed-size matrix
MatrixXf ProjPoints = MatrixXf::Zero(actualrowsize,3);
MatrixXf RawPoints = MatrixXf::Zero(actualrowsize,3);
ProjPoints.block(0,0,actualrowsize,3) = ProjPointsTemp.block(0,0,actualrowsize,3);
RawPoints.block(0,0,actualrowsize,3) = RawPointsTemp.block(0,0,actualrowsize,3);

// print out the vectors values to check the math
std::cout << A << endl << endl;
std::cout << B << endl << endl;
std::cout << AB << endl << endl;
std::cout << norm << endl << endl;

// output the selected regions of the RawPoints and check the validity of the codes for the projected points
for (int k=0; k < 20 ; ++k )  
{
std::cout << RawPoints.row(k) << endl << endl; 
}

for (int k=0; k < 20 ; ++k )  
{
std::cout << ProjPoints.row(k) << endl << endl; 
}

// setup initial matrix for sorting out projected points in different section, which can then be given weight for matric evaluation
MatrixXf UpPointsTemp = MatrixXf::Zero(actualrowsize,3);
MatrixXf MiddlePointsTemp = MatrixXf::Zero(actualrowsize,3);
MatrixXf LowPointsTemp = MatrixXf::Zero(actualrowsize,3);

int g1 = 0;
int g2 = 0;
int g3 = 0;

float UpZ = cone.height_;
float MiddleZ = cone.height_*2/3 ;
float LowZ = cone.height_*1/3 ;

// check the boundaries for each section, in this case it is the Z values to distinguish top, middle, low sections.
std::cout << cone.height_ << endl << endl;
std::cout << UpZ << endl << endl;
std::cout << MiddleZ<< endl << endl;
std::cout << LowZ << endl << endl;

// sort and assign points in the sections, using if function conditions to distinguish
for (int g = 0; g < actualrowsize; ++ g) 
{
    if ( ProjPoints(g,2)> MiddleZ &&   ProjPoints(g,2) <= UpZ  ) 
    {
        UpPointsTemp.row(g1) = ProjPoints.row(g);
        g1 = g1 + 1;
    }
    else if ( ProjPoints(g,2)> LowZ &&   ProjPoints(g,2) <= MiddleZ ) 
    {
        MiddlePointsTemp.row(g2) = ProjPoints.row(g);
        g2 = g2 + 1;
    }
    else if ( ProjPoints(g,2)>= 0 &&   ProjPoints(g,2) <= LowZ  ) 
    {
        LowPointsTemp.row(g3) = ProjPoints.row(g);
        g3 = g3 + 1;
    }
 }

 // get the actual row size in order to reshape the points into the final matrix with the correct row size for up, middle, and low sections 
int  actualg1 = g1;
int  actualg2 = g2;
int  actualg3 = g3;

std::cout << g1 << endl << endl;
std::cout << g2 << endl << endl;
std::cout << g3 << endl << endl;

// use the block operator in this case to store the filtered up middle low sections in a new set which has the right matrix size
// Important to note that, here the matrix is a dynamic matrix, the block function needs to be for dynamic too!!!
MatrixXf UpPoints = MatrixXf::Zero(actualg1,3);
MatrixXf MiddlePoints = MatrixXf::Zero(actualg2,3);
MatrixXf LowPoints = MatrixXf::Zero(actualg3,3);

UpPoints.block(0,0,actualg1,3) += UpPointsTemp.block(0,0,actualg1,3) ; 
MiddlePoints.block(0,0,actualg2,3) = MiddlePointsTemp.block(0,0,actualg2,3) ;
LowPoints.block(0,0,actualg3,3) = LowPointsTemp.block(0,0,actualg3,3) ;

// optional, output the points for specific sections
// std::cout << UpPoints << endl << endl;
// std::cout << MiddlePoints << endl << endl;
// std::cout << LowPoints << endl << endl;

}

};  // namespace Metrics
}  // namespace lcaster



#include <chrono>

int main() {
  using namespace lcaster;
  using namespace lcaster::Intersection;

  constexpr el_t HFOV = M_PI / 8;
  constexpr el_t HBIAS = -HFOV / 2;
  constexpr el_t VFOV = M_PI / 6;
  constexpr el_t VBIAS = -M_PI / 2;

  constexpr int NRings = 200;
  constexpr int NPoints = 200;
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
  Solutions<Dynamic> hit_height(rays.rays());

  World::DV world(ground, {cone});
  World::ObjectIdxs<Dynamic> object;
  // world.computeSolution(rays, solutions, hit_height, object);
  // ground.computeSolution(rays, solutions);
  cone.computeSolution(rays, solutions);
  (void)world;
  Points<Dynamic> points(rays.rays(), 3);
  computePoints(rays, solutions, points);

  // std::cout << points << "\n";

  PointCloud::Ptr cloud(new PointCloud);
  computePoints(rays, solutions, *cloud);

  Metrics::coneHeightMetric(cone, points);

  return 0;

  pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
  viewer.showCloud(cloud);
  while (!viewer.wasStopped()) {
  }

  return 0;
}
