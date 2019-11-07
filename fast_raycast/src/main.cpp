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

  // ENTER YOUR CODE HERE

  /*first to get each point XYZ from the points matrix */
  /* get the number of rows in the matrix points   */

  int rownumber = points.rows();

  /*then to setup a line with two points defined from the cone to be project on
  point A could be the vertix
  pint B could be the point at the the base of the cone, which is vertex +
  direction times height */
  Vector3e A = cone.vertex_;
  Vector3e B = cone.vertex_ + cone.height_ * cone.direction_;
  



  /* then use the projection equation to get the XYZ values from the solution
   * and assign into a new matrix called heightmetrix points      */
  // Computation of the coordinates of P
  
  


MatrixXf ProjPoints(rownumber,3);
// ProjPoints << MatrixXf::Zero(rownumber,3);


  // std::cout<< ProjPoints << "n/n/";

 Vector3e AB = B - A;
 float norm = AB.dot(AB);

int j = 0;
  for (int i = 0; i < points.rows(); ++i) {
    // The most inefficient version in the world (to be verified)
    Vector3e M = points.row(i).transpose();
   
    Vector3e AM = M - A;
    
    float dot = AB.dot(AM);
    float d1 = dot / norm;
    Vector3e AP = d1 * AB;
    Vector3e P = AP + A;

// here I tried to eliminate nan from the matrix by applying if condition for the P dot product of P to be a number, at least not nan


     if ( P.dot(P) > 0.0 ) {
          ProjPoints.row(j) = P.transpose();
          j = j + 1;

   }


  // At the end I need to resize the ProjPoints matrix to make remove the 0 0 0 at the end. 
  // Apparantly if you do not add vectors to the matrix, it initially is stored as 0 0 0 for each row

  }
// that is the size of the actual matrix with valid points in
std::cout << j << endl << endl;
int actualrowsize = j ;
std::cout << actualrowsize << endl << endl;


// now resize to the valid matrix size
ProjPoints.resize(actualrowsize,3);
  /* assign weight value on the points based on the section separation along the
   * cone projection line  */

// print out the ProjPoints matrix
std::cout << ProjPoints << endl << endl;
std::cout << ProjPoints.rows() << endl << endl;


std::cout << A << endl << endl;
std::cout << B << endl << endl;
std::cout << AB << endl << endl;
std::cout << norm << endl << endl;

std::cout << points.row(1).transpose() << endl << endl;
  

// this can print selected region of the rows
// for (int k=0; k < 200 ; ++k )  {
// std::cout << ProjPoints.row(k) << endl << endl;   }



// I want to filter out and see how the points are actually looking like
// I am not sure if the invalid calculation is due to the points themselves
// It is very important to know


// setup initial matrix for sorting out projected points in different section, which can then be given weight for matric evaluation



MatrixXf UpPoints(actualrowsize,3);
MatrixXf MiddlePoints(actualrowsize,3);
MatrixXf LowPoints(actualrowsize,3);

 // try plot the matrix in the XYZ cordinates, it should be basically the projected line
int g1 = 0;
int g2 = 0;
int g3 = 0;

float UpZ = cone.height_;
float MiddleZ = cone.height_*2/3 ;
float LowZ = cone.height_*1/3 ;

std::cout << cone.height_ << endl << endl;

std::cout << UpZ << endl << endl;

std::cout << MiddleZ<< endl << endl;

std::cout << LowZ << endl << endl;


std::cout << ProjPoints(1,2) << endl << endl;

// give weight and counts and bin the matrix so we can give it a weight, with stats on the ditribution on each sections
//  for (int g = 0; g < actualrowsize; ++ g) {
//         if ( ProjPoints(g,2)> MiddleZ &&   ProjPoints(g,2) <= UpZ  ) {
//           UpPoints.row(g1) = ProjPoints.row(g);
//            g1 = g1 + 1;
//         }
//         else if ( ProjPoints(g,2)> LowZ &&   ProjPoints(g,2) <= MiddleZ ) {
//            MiddlePoints.row(g2) = ProjPoints.row(g);
//            g2 = g2 + 1;
//         }

//         else if ( ProjPoints(g,2)>= 0 &&   ProjPoints(g,2) <= LowZ  ) {
//            LowPoints.row(g1) = ProjPoints.row(g);
//            g3 = g3 + 1;
//         
//         }
//     }
  
      for (int g = 0; g < actualrowsize; ++ g) {
         if ( ProjPoints(g,2)> MiddleZ &&   ProjPoints(g,2) <= UpZ  ) {
            UpPoints.block<1,3>(g1,0) = ProjPoints.row(g);
            g1 = g1 + 1;
         }
         else if ( ProjPoints(g,2)> LowZ &&   ProjPoints(g,2) <= MiddleZ ) {
            MiddlePoints.block<1,3>(g1,0) = ProjPoints.row(g);
            g2 = g2 + 1;
         }

         else if ( ProjPoints(g,2)>= 0 &&   ProjPoints(g,2) <= LowZ  ) {
            LowPoints.block<1,3>(g1,0) = ProjPoints.row(g);
            g3 = g3 + 1;
         
         }
       
  }

 // get the actual row size in order to resize the sectional matrix for up, middle, and low sections 
int  actualg1 = g1;
int  actualg2 = g2;
int  actualg3 = g3;

std::cout << g1 << endl << endl;
std::cout << g2 << endl << endl;
std::cout << g3 << endl << endl;


// plot a 1-D line with the weights


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
