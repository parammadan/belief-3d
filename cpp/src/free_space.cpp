// Free-space log-likelihood kernel for the SE(3) particle belief (M7).
//
// Exact port of tracking/belief/particle.py::PoseBelief.free_space_log_lik (the NumPy CPU
// reference). Per particle n and model surface sample m: transform the sample into the camera,
// keep camera-facing samples in front of the camera and inside the image, look up the observed
// depth, and penalise pixels where the camera saw *behind* the hypothesised surface. Each sample
// is weighted by the pixels it covers. Rounding uses nearbyint (round-half-to-even, like
// np.round), so pixel lookups match the reference exactly.
#include <cmath>
#include <cstdint>

namespace belief3d {

void free_space_log_lik(const double* R, const double* x, std::int64_t N,
                        const double* S, const double* Nrm, std::int64_t M,
                        const double* T_CW,
                        double fx, double fy, double cx, double cy, std::int64_t width, std::int64_t height,
                        const float* depth, double area_per_sample, double s, double margin,
                        double weight, double temper, double* out) {
  const double* Rcw = T_CW;          // row-major 4x4: rows 0..2, cols 0..2 rotation, col 3 translation
  for (std::int64_t n = 0; n < N; ++n) {
    const double* Rn = R + 9 * n;
    const double* xn = x + 3 * n;
    double RC[9], tC[3];
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        RC[3 * i + j] = Rcw[4 * i + 0] * Rn[0 * 3 + j] + Rcw[4 * i + 1] * Rn[1 * 3 + j] + Rcw[4 * i + 2] * Rn[2 * 3 + j];
      }
      tC[i] = Rcw[4 * i + 0] * xn[0] + Rcw[4 * i + 1] * xn[1] + Rcw[4 * i + 2] * xn[2] + Rcw[4 * i + 3];
    }
    double acc = 0.0;
    for (std::int64_t m = 0; m < M; ++m) {
      const double* sm = S + 3 * m;
      const double* nm = Nrm + 3 * m;
      double p[3], q[3];
      for (int i = 0; i < 3; ++i) {
        p[i] = RC[3 * i] * sm[0] + RC[3 * i + 1] * sm[1] + RC[3 * i + 2] * sm[2] + tC[i];
        q[i] = RC[3 * i] * nm[0] + RC[3 * i + 1] * nm[1] + RC[3 * i + 2] * nm[2];
      }
      const double nd = q[0] * p[0] + q[1] * p[1] + q[2] * p[2];
      const double z = p[2];
      if (!(nd < 0.0) || !(z > 1e-6)) continue;
      const double u = fx * p[0] / z + cx;
      const double v = fy * p[1] / z + cy;
      // Reject far-off projections before the integer cast (out-of-range casts are UB).
      if (!(u > -1.0 && u < static_cast<double>(width) + 1.0 && v > -1.0 && v < static_cast<double>(height) + 1.0)) continue;
      const std::int64_t ui = static_cast<std::int64_t>(std::nearbyint(u));
      const std::int64_t vi = static_cast<std::int64_t>(std::nearbyint(v));
      if (ui < 0 || ui >= width || vi < 0 || vi >= height) continue;
      const double zo = static_cast<double>(depth[vi * width + ui]);
      if (!(zo > 0.0)) continue;
      const double viol = zo - z - margin;
      if (!(viol > 0.0)) continue;
      const double cosang = std::fabs(nd) / std::fmax(std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]), 1e-9);
      const double pix = area_per_sample * cosang * (fx / z) * (fy / z);
      acc += pix * std::fmin(viol / s, 3.0);
    }
    out[n] = -weight * acc / temper;
  }
}

}  // namespace belief3d
