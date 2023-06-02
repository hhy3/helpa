#pragma once

#if defined(__AVX512F__)

#include "helpa/l2.hpp"
#include "helpa/platform/avx512/avx512_utils.hpp"
#include "helpa/ref/l2_ref.hpp"

#include "helpa/types.hpp"

namespace helpa {

inline float l2_fp32_fp32(const float *x, const float *y, const int32_t d) {
  int32_t da = d / 16 * 16;
  return l2a_fp32_fp32(x, y, da) + l2_fp32_fp32_ref(x + da, y + da, d - da);
}

inline float l2_fp32_fp16(const float *x, const fp16 *y, const int32_t d) {
  int32_t da = d / 16 * 16;
  return l2a_fp32_fp16(x, y, da) + l2_fp32_fp16_ref(x + da, y + da, d - da);
}

inline float l2_fp16_fp16(const fp16 *x, const fp16 *y, const int32_t d) {
  int32_t da = d / 32 * 32;
  return l2a_fp16_fp16(x, y, da) + l2_fp16_fp16_ref(x + da, y + da, d - da);
}

inline float l2_fp32_bf16(const float *x, const bf16 *y, const int32_t d) {
  int32_t da = d / 16 * 16;
  return l2a_fp32_bf16(x, y, da) + l2_fp32_bf16_ref(x + da, y + da, d - da);
}

inline float l2_bf16_bf16(const bf16 *x, const bf16 *y, const int32_t d) {
  int32_t da = d / 32 * 32;
  return l2a_bf16_bf16(x, y, da) + l2_bf16_bf16_ref(x + da, y + da, d - da);
}

inline int32_t l2_s8_s8(const int8_t *x, const int8_t *y, const int32_t d) {
  int32_t da = d / 64 * 64;
  return l2a_s8_s8(x, y, da) + l2_s8_s8_ref(x + da, y + da, d - da);
}

inline int32_t l2_u8_u8(const uint8_t *x, const uint8_t *y, const int32_t d) {
  int32_t da = d / 64 * 64;
  return l2a_u8_u8(x, y, da) + l2_u8_u8_ref(x + da, y + da, d - da);
}

inline float l2a_fp32_fp32(const float *x, const float *y, const int32_t d) {
  __m256 sum = _mm256_setzero_ps();
  for (int32_t i = 0; i < d; i += 8) {
    auto xx = _mm256_loadu_ps(x + i);
    auto yy = _mm256_loadu_ps(y + i);
    auto t = _mm256_sub_ps(xx, yy);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(t, t));
  }
  return reduce_add_f32x8(sum);
}

inline float l2a_fp32_fp16(const float *x, const fp16 *y, const int32_t d) {
  __m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
  for (int i = 0; i < d; i += 16) {
    {
      auto xx = _mm256_loadu_ps(x + i);
      auto zz = _mm_loadu_si128((__m128i *)(y + i));
      auto yy = _mm256_cvtph_ps(zz);
      auto t = _mm256_sub_ps(xx, yy);
      sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(t, t));
    }
    {
      auto xx = _mm256_loadu_ps(x + i + 8);
      auto zz = _mm_loadu_si128((__m128i *)(y + i + 8));
      auto yy = _mm256_cvtph_ps(zz);
      auto t = _mm256_sub_ps(xx, yy);
      sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(t, t));
    }
  }
  sum1 = _mm256_add_ps(sum1, sum2);
  return reduce_add_f32x8(sum1);
}

inline float l2a_fp16_fp16(const fp16 *x, const fp16 *y, const int32_t d) {
  __m256 sum = _mm256_setzero_ps();
  for (int i = 0; i < d; i += 8) {
    auto xxx = _mm_loadu_si128((__m128i *)(x + i));
    auto xx = _mm256_cvtph_ps(xxx);
    auto zz = _mm_loadu_si128((__m128i *)(y + i));
    auto yy = _mm256_cvtph_ps(zz);
    auto t = _mm256_sub_ps(xx, yy);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(t, t));
  }
  return reduce_add_f32x8(sum);
}

inline float l2a_fp32_bf16(const float *x, const bf16 *y, const int32_t d) {
  __m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
  for (int i = 0; i < d; i += 16) {
    {
      auto xx = _mm256_loadu_ps(x + i);
      auto zz = _mm_loadu_si128((__m128i *)(y + i));
      auto yy = _mm256_cvtepu16_epi32(zz);
      yy = _mm256_slli_epi32(yy, 16);
      auto t = _mm256_sub_ps(xx, (__m256)yy);
      sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(t, t));
    }
    {
      auto xx = _mm256_loadu_ps(x + i + 8);
      auto zz = _mm_loadu_si128((__m128i *)(y + i + 8));
      auto yy = _mm256_cvtepu16_epi32(zz);
      yy = _mm256_slli_epi32(yy, 16);
      auto t = _mm256_sub_ps(xx, (__m256)yy);
      sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(t, t));
    }
  }
  sum1 = _mm256_add_ps(sum1, sum2);
  return reduce_add_f32x8(sum1);
}

inline float l2a_bf16_bf16(const bf16 *x, const bf16 *y, const int32_t d) {
  __m256 sum = _mm256_setzero_ps();
  for (int i = 0; i < d; i += 8) {
    auto xxx = _mm_loadu_si128((__m128i *)(x + i));
    auto xx = _mm256_cvtepu16_epi32(xxx);
    xx = _mm256_slli_epi32(xx, 16);
    auto zz = _mm_loadu_si128((__m128i *)(y + i));
    auto yy = _mm256_cvtepu16_epi32(zz);
    yy = _mm256_slli_epi32(yy, 16);
    auto t = _mm256_sub_ps((__m256)xx, (__m256)yy);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(t, t));
  }
  return reduce_add_f32x8(sum);
}

// x[i], y[i] in [-63, 63]
inline int32_t l2a_s8_s8(const int8_t *x, const int8_t *y, const int32_t d) {
  __m256i sum1 = _mm256_setzero_si256(), sum2 = _mm256_setzero_si256();
  for (int i = 0; i < d; i += 64) {
    {
      auto xx = _mm256_loadu_si256((__m256i *)(x + i));
      auto yy = _mm256_loadu_si256((__m256i *)(y + i));
      auto t = _mm256_sub_epi8(xx, yy);
      t = _mm256_abs_epi8(t);
      auto tmp = _mm256_maddubs_epi16(t, t);
      sum1 = _mm256_add_epi32(
          sum1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 0)));
      sum1 = _mm256_add_epi32(
          sum1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 1)));
    }
    {
      auto xx = _mm256_loadu_si256((__m256i *)(x + i + 32));
      auto yy = _mm256_loadu_si256((__m256i *)(y + i + 32));
      auto t = _mm256_sub_epi8(xx, yy);
      t = _mm256_abs_epi8(t);
      auto tmp = _mm256_maddubs_epi16(t, t);
      sum2 = _mm256_add_epi32(
          sum2, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 0)));
      sum2 = _mm256_add_epi32(
          sum2, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 1)));
    }
  }
  sum1 = _mm256_add_epi32(sum1, sum2);
  return reduce_add_i32x8(sum1);
}

// x[i], y[i] in [0, 127]
inline int32_t l2a_u8_u8(const uint8_t *x, const uint8_t *y, const int32_t d) {
  __m256i sum1 = _mm256_setzero_si256(), sum2 = _mm256_setzero_si256();
  for (int i = 0; i < d; i += 64) {
    {
      auto xx = _mm256_loadu_si256((__m256i *)(x + i));
      auto yy = _mm256_loadu_si256((__m256i *)(y + i));
      auto t = _mm256_sub_epi8(xx, yy);
      t = _mm256_abs_epi8(t);
      auto tmp = _mm256_maddubs_epi16(t, t);
      sum1 = _mm256_add_epi32(
          sum1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 0)));
      sum1 = _mm256_add_epi32(
          sum1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 1)));
    }
    {
      auto xx = _mm256_loadu_si256((__m256i *)(x + i + 32));
      auto yy = _mm256_loadu_si256((__m256i *)(y + i + 32));
      auto t = _mm256_sub_epi8(xx, yy);
      t = _mm256_abs_epi8(t);
      auto tmp = _mm256_maddubs_epi16(t, t);
      sum2 = _mm256_add_epi32(
          sum2, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 0)));
      sum2 = _mm256_add_epi32(
          sum2, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(tmp, 1)));
    }
  }
  sum1 = _mm256_add_epi32(sum1, sum2);
  return reduce_add_i32x8(sum1);
}

inline int32_t l2a_u4_u4(const uint8_t *x, const uint8_t *y, const int32_t d) {
  __m256i sum1 = _mm256_setzero_si256(), sum2 = _mm256_setzero_si256();
  __m256i mask = _mm256_set1_epi8(0xf);
  for (int i = 0; i < d; i += 64) {
    auto xx = _mm256_loadu_si256((__m256i *)(x + i / 2));
    auto yy = _mm256_loadu_si256((__m256i *)(y + i / 2));
    auto xx1 = _mm256_and_si256(xx, mask);
    auto xx2 = _mm256_and_si256(_mm256_srli_epi16(xx, 4), mask);
    auto yy1 = _mm256_and_si256(yy, mask);
    auto yy2 = _mm256_and_si256(_mm256_srli_epi16(yy, 4), mask);
    auto d1 = _mm256_sub_epi8(xx1, yy1);
    auto d2 = _mm256_sub_epi8(xx2, yy2);
    d1 = _mm256_abs_epi8(d1);
    d2 = _mm256_abs_epi8(d2);
    sum1 = _mm256_add_epi16(sum1, _mm256_maddubs_epi16(d1, d1));
    sum2 = _mm256_add_epi16(sum2, _mm256_maddubs_epi16(d2, d2));
  }
  sum1 = _mm256_add_epi32(sum1, sum2);
  return reduce_add_i16x16(sum1);
}

} // namespace helpa

#endif
