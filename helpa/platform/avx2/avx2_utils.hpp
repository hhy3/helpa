#pragma once

#if !defined(__AVX512F__) && defined(__AVX2__)

#include <immintrin.h>

#include "helpa/common.hpp"

namespace helpa {

HELPA_INLINE inline float reduce_add_f32x8(__m256 x) {
  auto sumh =
      _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
  auto tmp1 = _mm_add_ps(sumh, _mm_movehl_ps(sumh, sumh));
  auto tmp2 = _mm_add_ps(tmp1, _mm_movehdup_ps(tmp1));
  return _mm_cvtss_f32(tmp2);
}

HELPA_INLINE inline int32_t reduce_add_i32x8(__m256i x) {
  auto sumh =
      _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x, 1));
  auto tmp2 = _mm_hadd_epi32(sumh, sumh);
  return _mm_extract_epi32(tmp2, 0) + _mm_extract_epi32(tmp2, 1);
}

HELPA_INLINE inline int32_t reduce_add_i16x16(__m256i x) {
  auto sumh = _mm_add_epi16(_mm256_extracti128_si256(x, 0),
                            _mm256_extracti128_si256(x, 1));
  auto tmp = _mm256_cvtepi16_epi32(sumh);
  auto sumhh = _mm_add_epi32(_mm256_extracti128_si256(tmp, 0),
                             _mm256_extracti128_si256(tmp, 1));
  auto tmp2 = _mm_hadd_epi32(sumhh, sumhh);
  return _mm_extract_epi32(tmp2, 0) + _mm_extract_epi32(tmp2, 1);
}

HELPA_INLINE inline __m256i cvti4x32_i8x32(__m128i x) {
  auto mask = _mm_set1_epi8(0x0f);
  auto lo = _mm_and_si128(x, mask);
  auto hi = _mm_and_si128(_mm_srli_epi16(x, 4), mask);
  auto loo = _mm256_cvtepu8_epi16(lo);
  auto hii = _mm256_cvtepu8_epi16(hi);
  hii = _mm256_slli_si256(hii, 1);
  auto ret = _mm256_or_si256(loo, hii);
  ret = _mm256_slli_epi64(ret, 3);
  return ret;
}

} // namespace helpa

#endif
