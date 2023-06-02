#pragma once

#if defined(__AVX2__)

#include "helpa/dot.hpp"
#include "helpa/platform/avx2/avx2_utils.hpp"
#include "helpa/ref/dot_ref.hpp"

#include "helpa/types.hpp"

namespace helpa {

inline float dot_fp32_fp32(const float *x, const float *y, const int32_t d) {
  int32_t da = d / 16 * 16;
  return dotu_fp32_fp32(x, y, da) + dot_fp32_fp32_ref(x + da, y + da, d - da);
}

inline float dot_fp32_fp16(const float *x, const fp16 *y, const int32_t d) {
  int32_t da = d / 16 * 16;
  return dotu_fp32_fp16(x, y, da) + dot_fp32_fp16_ref(x + da, y + da, d - da);
}

inline float dot_fp16_fp16(const fp16 *x, const fp16 *y, const int32_t d) {
  int32_t da = d / 32 * 32;
  return dotu_fp16_fp16(x, y, da) + dot_fp16_fp16_ref(x + da, y + da, d - da);
}

inline float dot_fp32_bf16(const float *x, const bf16 *y, const int32_t d) {
  int32_t da = d / 16 * 16;
  return dotu_fp32_bf16(x, y, da) + dot_fp32_bf16_ref(x + da, y + da, d - da);
}

inline float dot_bf16_bf16(const bf16 *x, const bf16 *y, const int32_t d) {
  int32_t da = d / 32 * 32;
  return dotu_bf16_bf16(x, y, da) + dot_bf16_bf16_ref(x + da, y + da, d - da);
}

inline int32_t dot_s8_s8(const int8_t *x, const int8_t *y, const int32_t d) {
  int32_t da = d / 64 * 64;
  return dotu_s8_s8(x, y, da) + dot_s8_s8_ref(x + da, y + da, d - da);
}

inline int32_t dot_u8_u8(const uint8_t *x, const uint8_t *y, const int32_t d) {
  int32_t da = d / 64 * 64;
  return dotu_u8_u8(x, y, da) + dot_u8_u8_ref(x + da, y + da, d - da);
}

inline float dotu_fp32_fp32(const float *x, const float *y, const int32_t d) {
  __m256 sum = _mm256_setzero_ps();
  for (int32_t i = 0; i < d; i += 8) {
    auto xx = _mm256_loadu_ps(x + i);
    auto yy = _mm256_loadu_ps(y + i);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(xx, yy));
  }
  return reduce_add_f32x8(sum);
}

inline float dotu_fp32_fp16(const float *x, const fp16 *y, const int32_t d) {
  __m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
  for (int i = 0; i < d; i += 16) {
    {
      auto xx = _mm256_loadu_ps(x + i);
      auto zz = _mm_loadu_si128((__m128i *)(y + i));
      auto yy = _mm256_cvtph_ps(zz);
      sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(xx, yy));
    }
    {
      auto xx = _mm256_loadu_ps(x + i + 8);
      auto zz = _mm_loadu_si128((__m128i *)(y + i + 8));
      auto yy = _mm256_cvtph_ps(zz);
      sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(xx, yy));
    }
  }
  sum1 = _mm256_add_ps(sum1, sum2);
  return reduce_add_f32x8(sum1);
}

inline float dotu_fp16_fp16(const fp16 *x, const fp16 *y, const int32_t d) {
  __m256 sum = _mm256_setzero_ps();
  for (int i = 0; i < d; i += 8) {
    auto xxx = _mm_loadu_si128((__m128i *)(x + i));
    auto xx = _mm256_cvtph_ps(xxx);
    auto zz = _mm_loadu_si128((__m128i *)(y + i));
    auto yy = _mm256_cvtph_ps(zz);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(xx, yy));
  }
  return reduce_add_f32x8(sum);
}

inline float dotu_fp32_bf16(const float *x, const bf16 *y, const int32_t d) {
  __m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
  for (int i = 0; i < d; i += 16) {
    {
      auto xx = _mm256_loadu_ps(x + i);
      auto zz = _mm_loadu_si128((__m128i *)(y + i));
      auto yy = _mm256_cvtepu16_epi32(zz);
      yy = _mm256_slli_epi32(yy, 16);
      sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(xx, (__m256)yy));
    }
    {
      auto xx = _mm256_loadu_ps(x + i + 8);
      auto zz = _mm_loadu_si128((__m128i *)(y + i + 8));
      auto yy = _mm256_cvtepu16_epi32(zz);
      yy = _mm256_slli_epi32(yy, 16);
      sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(xx, (__m256)yy));
    }
  }
  sum1 = _mm256_add_ps(sum1, sum2);
  return reduce_add_f32x8(sum1);
}

inline float dotu_bf16_bf16(const bf16 *x, const bf16 *y, const int32_t d) {
  __m256 sum = _mm256_setzero_ps();
  for (int i = 0; i < d; i += 8) {
    auto xxx = _mm_loadu_si128((__m128i *)(x + i));
    auto xx = _mm256_cvtepu16_epi32(xxx);
    xx = _mm256_slli_epi32(xx, 16);
    auto zz = _mm_loadu_si128((__m128i *)(y + i));
    auto yy = _mm256_cvtepu16_epi32(zz);
    yy = _mm256_slli_epi32(yy, 16);
    sum = _mm256_add_ps(sum, _mm256_mul_ps((__m256)xx, (__m256)yy));
  }
  return reduce_add_f32x8(sum);
}

inline int32_t dotu_s8_s8(const int8_t *x, const int8_t *y, const int32_t d) {
  return dot_s8_s8(x, y, d);
}

inline int32_t dotu_u8_u8(const uint8_t *x, const uint8_t *y, const int32_t d) {
  return dot_u8_u8(x, y, d);
}

} // namespace helpa

#endif
