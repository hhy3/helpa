#pragma once

#include <emmintrin.h>

#include "helpa/common.hpp"

#if defined(HELPA_USE_X86)

#include <immintrin.h>

#include "helpa/data_convert.hpp"
#include "helpa/ref/data_convert_ref.hpp"
#include "helpa/types.hpp"

namespace helpa {

inline void
fp32_to_fp16(const float* from, fp16* to, size_t size) {
    size_t sa = size / 16 * 16;
    fp32_to_fp16_align(from, to, size);
    fp32_to_fp16_ref(from + sa, to + sa, size - sa);
}

inline void
fp32_to_bf16(const float* from, bf16* to, size_t size) {
    size_t sa = size / 16 * 16;
    fp32_to_bf16_align(from, to, size);
    fp32_to_bf16_ref(from + sa, to + sa, size - sa);
}

inline void
fp16_to_fp32(const fp16* from, float* to, size_t size) {
    size_t sa = size / 16 * 16;
    fp16_to_fp32_align(from, to, size);
    fp16_to_fp32_ref(from + sa, to + sa, size - sa);
}

inline void
bf16_to_fp32(const bf16* from, float* to, size_t size) {
    size_t sa = size / 16 * 16;
    bf16_to_fp32_align(from, to, size);
    bf16_to_fp32_ref(from + sa, to + sa, size - sa);
}

inline void
fp32_to_fp16_align(const float* from, fp16* to, size_t size) {
#if defined(HELPA_USE_AVX512)
    for (size_t i = 0; i < size; i += 16) {
        auto xx = _mm512_loadu_ps(from + i);
        auto yy = _mm512_cvtps_ph(xx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm256_storeu_si256((__m256i*)(to + i), yy);
    }
#else
    for (size_t i = 0; i < size; i += 8) {
        auto xx = _mm256_loadu_ps(from + i);
        auto yy = _mm256_cvtps_ph(xx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i*)(to + i), yy);
    }
#endif
}

inline void
fp32_to_bf16_align(const float* from, bf16* to, size_t size) {
#if defined(HELPA_USE_AVX512)
    for (size_t i = 0; i < size; i += 16) {
        auto xx = _mm512_loadu_si512(from + i);
        xx = _mm512_srli_epi32(xx, 16);
        auto yy = _mm512_cvtepi32_epi16(xx);
        _mm256_storeu_si256((__m256i*)(to + i), yy);
    }
#else
    for (size_t i = 0; i < size; i += 8) {
        auto xx = _mm256_loadu_si256((__m256i*)(from + i));
        xx = _mm256_srli_epi32(xx, 16);
        auto yy = _mm256_cvtepi32_epi16(xx);
        _mm_storeu_si128((__m128i*)(to + i), yy);
    }
#endif
}

inline void
fp16_to_fp32_align(const fp16* from, float* to, size_t size) {
#if defined(HELPA_USE_AVX512)
    for (size_t i = 0; i < size; i += 16) {
        auto xx = _mm256_loadu_si256((__m256i*)(from + i));
        auto yy = _mm512_cvtph_ps(xx);
        _mm512_storeu_ps(to + i, yy);
    }
#else
    for (size_t i = 0; i < size; i += 8) {
        auto xx = _mm_loadu_si128((__m128i*)(from + i));
        auto yy = _mm256_cvtph_ps(xx);
        _mm256_storeu_ps(to + i, yy);
    }
#endif
}

inline void
bf16_to_fp32_align(const bf16* from, float* to, size_t size) {
#if defined(HELPA_USE_AVX512)
    for (size_t i = 0; i < size; i += 16) {
        auto xx = _mm256_loadu_si256((__m256i*)(from + i));
        auto yy = _mm512_cvtepu16_epi32(xx);
        yy = _mm512_slli_epi32(yy, 16);
        _mm512_storeu_si512(to + i, yy);
    }
#else
    for (size_t i = 0; i < size; i += 8) {
        auto xx = _mm_loadu_si128((__m128i*)(from + i));
        auto yy = _mm256_cvtepu16_epi32(xx);
        yy = _mm256_slli_epi32(yy, 16);
        _mm256_storeu_si256((__m256i*)(to + i), yy);
    }
#endif
}

}  // namespace helpa

#endif
