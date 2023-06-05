#pragma once

#include "helpa/types.hpp"

namespace helpa {

inline float l2_fp32_fp32_ref(const float *x, const float *y, const int32_t d) {
  auto ans = 0.0f;
  for (int32_t i = 0; i < d; ++i) {
    auto d = x[i] - y[i];
    ans += d * d;
  }
  return ans;
}

inline float l2_fp32_fp16_ref(const float *x, const fp16 *y, const int32_t d) {
  auto ans = 0.0f;
  for (int32_t i = 0; i < d; ++i) {
    auto d = x[i] - float(y[i]);
    ans += d * d;
  }
  return ans;
}

inline float l2_fp16_fp16_ref(const fp16 *x, const fp16 *y, const int32_t d) {
  auto ans = 0.0f;
  for (int32_t i = 0; i < d; ++i) {
    auto d = float(x[i]) - float(y[i]);
    ans += d * d;
  }
  return ans;
}

inline float l2_fp32_bf16_ref(const float *x, const bf16 *y, const int32_t d) {
  float ans = 0.0f;
  for (int32_t i = 0; i < d; ++i) {
    auto d = x[i] - float(y[i]);
    ans += d * d;
  }
  return ans;
}

inline float l2_bf16_bf16_ref(const bf16 *x, const bf16 *y, const int32_t d) {
  float ans = 0.0f;
  for (int32_t i = 0; i < d; ++i) {
    auto d = float(x[i]) - float(y[i]);
    ans += d * d;
  }
  return ans;
}

inline int32_t l2_u8_u8_ref(const uint8_t *x, const uint8_t *y,
                            const int32_t d) {
  int32_t ans = 0;
  for (int32_t i = 0; i < d; ++i) {
    auto d = int32_t(x[i]) - int32_t(y[i]);
    ans += d * d;
  }
  return ans;
}

inline int32_t l2_u4_u4_ref(const uint8_t *x, const uint8_t *y, const int32_t d) {
  int32_t ans = 0;
  for (int32_t i = 0; i < d; ++i) {
    int32_t xx = x[i / 2] >> ((i & 1) * 4) & 15;
    int32_t yy = y[i / 2] >> ((i & 1) * 4) & 15;
    auto d = xx - yy;
    ans += d * d;
  }
  return ans;
}

} // namespace helpa

