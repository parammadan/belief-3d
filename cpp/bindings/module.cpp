// pybind11 bindings: belief3d_cpp.free_space_log_lik(...) -> np.ndarray (N,)
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

namespace belief3d {
void free_space_log_lik(const double* R, const double* x, std::int64_t N, const double* S, const double* Nrm,
                        std::int64_t M, const double* T_CW, double fx, double fy, double cx, double cy,
                        std::int64_t width, std::int64_t height, const float* depth, double area_per_sample,
                        double s, double margin, double weight, double temper, double* out);
}

using cd = py::array_t<double, py::array::c_style | py::array::forcecast>;
using cf = py::array_t<float, py::array::c_style | py::array::forcecast>;

static py::array_t<double> free_space(cd R, cd x, cd S, cd Nrm, cd T_CW, double fx, double fy, double cx, double cy,
                                      cf depth, double area, double s, double margin, double weight, double temper) {
  if (R.ndim() != 3 || R.shape(1) != 3 || R.shape(2) != 3) throw std::invalid_argument("R must be (N,3,3)");
  if (x.ndim() != 2 || x.shape(1) != 3 || x.shape(0) != R.shape(0)) throw std::invalid_argument("x must be (N,3)");
  if (S.ndim() != 2 || S.shape(1) != 3 || Nrm.shape(0) != S.shape(0)) throw std::invalid_argument("S, Nrm must be (M,3)");
  if (T_CW.ndim() != 2 || T_CW.shape(0) != 4 || T_CW.shape(1) != 4) throw std::invalid_argument("T_CW must be 4x4");
  if (depth.ndim() != 2) throw std::invalid_argument("depth must be (H,W)");
  const auto N = R.shape(0);
  py::array_t<double> out(N);
  {
    py::gil_scoped_release release;
    belief3d::free_space_log_lik(R.data(), x.data(), N, S.data(), Nrm.data(), S.shape(0), T_CW.data(), fx, fy, cx, cy,
                                 depth.shape(1), depth.shape(0), depth.data(), area, s, margin, weight, temper,
                                 out.mutable_data());
  }
  return out;
}

PYBIND11_MODULE(belief3d_cpp, m) {
  m.doc() = "BELIEF-3D C++ kernels (M7): exact ports of profiled NumPy hot paths";
  m.def("free_space_log_lik", &free_space, "Free-space log-likelihood per particle (exact port of the NumPy reference)");
}
