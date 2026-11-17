// A float32 tanh the compiler vectorises, for platforms without a vector library.
//
// Cephes' tanhf: a polynomial in x^2 below 0.625, 1 - 2 / (exp(2|x|) + 1) above, with Cephes'
// expf (range reduction by log 2, a degree-6 polynomial, the power of two by the exponent bits).
// No branches, so the loop over an array becomes SIMD under -O3 with -march=native; the error is
// within two units in the last place of float32 (checked against the C library over [-10, 10]).
// Apple builds use Accelerate's vvtanhf instead.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace fast_math {

inline float expf_approx(float x) {
    x = x < -87.0f ? -87.0f : (x > 88.0f ? 88.0f : x);
    const float fx = std::floor(x * 1.44269504088896341f + 0.5f);
    x -= fx * 0.693359375f;
    x -= fx * -2.12194440e-4f;
    const float z = x * x;
    float y = 1.9875691500e-4f;
    y = y * x + 1.3981999507e-3f;
    y = y * x + 8.3334519073e-3f;
    y = y * x + 4.1665795894e-2f;
    y = y * x + 1.6666665459e-1f;
    y = y * x + 5.0000001201e-1f;
    y = y * z + x + 1.0f;
    const uint32_t bits = (uint32_t)((int32_t)fx + 127) << 23;
    float scale;
    std::memcpy(&scale, &bits, sizeof scale);
    return y * scale;
}

inline float tanhf_approx(float x) {
    const float ax = std::fabs(x);
    const float z = x * x;
    float p = -5.70498872745e-3f;
    p = p * z + 2.06390887954e-2f;
    p = p * z - 5.37397155531e-2f;
    p = p * z + 1.33314422036e-1f;
    p = p * z - 3.33332819422e-1f;
    const float small = p * z * x + x;
    const float e = expf_approx(2.0f * (ax < 9.0f ? ax : 9.0f));
    const float large = std::copysign(1.0f - 2.0f / (e + 1.0f), x);
    return ax < 0.625f ? small : large;
}

}  // namespace fast_math
