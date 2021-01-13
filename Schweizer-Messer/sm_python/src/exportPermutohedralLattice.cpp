#include <numpy_eigen/boost_python_headers.hpp>
#include "sm/permutohedral.h"

using namespace sm::permutohedral;

Eigen::MatrixXf filter(const Permutohedral& ph, const Eigen::MatrixXf& v, int start)
{
    return ph.compute(v, false, start);
}
void exportPermutohedralLattice() {
  using namespace boost::python;

  class_<Permutohedral>("Permutohedral", init<>())
    .def("init", &Permutohedral::init)
    .def("get_lattice_size", &Permutohedral::getLatticeSize)
    .def("filter", &filter);
}