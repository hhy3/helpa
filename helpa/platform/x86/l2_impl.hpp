#pragma once

#include <immintrin.h>
#if defined(__AVX2__)

#include "helpa/l2.hpp"
#include "helpa/platform/x86/utils.hpp"
#include "helpa/ref/l2_ref.hpp"
#include "helpa/types.hpp"

namespace helpa {

inline float
l2_fp32_fp32(const float* x, const float* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return l2a_fp32_fp32(x, y, da) + l2_fp32_fp32_ref(x + da, y + da, d - da);
}

inline float
l2_fp32_fp16(const float* x, const fp16* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return l2a_fp32_fp16(x, y, da) + l2_fp32_fp16_ref(x + da, y + da, d - da);
}

inline float
l2_fp16_fp16(const fp16* x, const fp16* y, const int32_t d) {
    int32_t da = d / 32 * 32;
    return l2a_fp16_fp16(x, y, da) + l2_fp16_fp16_ref(x + da, y + da, d - da);
}

inline float
l2_fp32_bf16(const float* x, const bf16* y, const int32_t d) {
    int32_t da = d / 16 * 16;
    return l2a_fp32_bf16(x, y, da) + l2_fp32_bf16_ref(x + da, y + da, d - da);
}

inline float
l2_bf16_bf16(const bf16* x, const bf16* y, const int32_t d) {
    int32_t da = d / 32 * 32;
    return l2a_bf16_bf16(x, y, da) + l2_bf16_bf16_ref(x + da, y + da, d - da);
}

inline int32_t
l2_u7_u7(const uint8_t* x, const uint8_t* y, const int32_t d) {
    int32_t da = d / 64 * 64;
    return l2a_u7_u7(x, y, da) + l2_u7_u7_ref(x + da, y + da, d - da);
}

inline int32_t
l2_s8_s8(const int8_t* x, const int8_t* y, const int32_t d) {
    int32_t da = d / 64 * 64;
    return l2a_s8_s8(x, y, da) + l2_s8_s8_ref(x + da, y + da, d - da);
}

inline float
l2a_fp32_fp32(const float* x, const float* y, const int32_t d) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (int32_t i = 0; i < d; i += 16) {
        auto xx = _mm512_loadu_ps(x + i);
        auto yy = _mm512_loadu_ps(y + i);
        auto t = _mm512_sub_ps(xx, yy);
        sum = _mm512_fmadd_ps(t, t, sum);
    }
    return reduce_add_f32x16(sum);
#else
    __m256 sum = _mm256_setzero_ps();
    for (int32_t i = 0; i < d; i += 8) {
        auto xx = _mm256_loadu_ps(x + i);
        auto yy = _mm256_loadu_ps(y + i);
        auto t = _mm256_sub_ps(xx, yy);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(t, t));
    }
    return reduce_add_f32x8(sum);
#endif
}

inline float
l2a_fp32_fp16(const float* x, const fp16* y, const int32_t d) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        auto xx = _mm512_loadu_ps(x + i);
        auto zz = _mm256_loadu_si256((__m256i*)(y + i));
        auto yy = _mm512_cvtph_ps(zz);
        auto t = _mm512_sub_ps(xx, yy);
        sum = _mm512_fmadd_ps(t, t, sum);
    }
    return reduce_add_f32x16(sum);
#else
    __m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        {
            auto xx = _mm256_loadu_ps(x + i);
            auto zz = _mm_loadu_si128((__m128i*)(y + i));
            auto yy = _mm256_cvtph_ps(zz);
            auto t = _mm256_sub_ps(xx, yy);
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(t, t));
        }
        {
            auto xx = _mm256_loadu_ps(x + i + 8);
            auto zz = _mm_loadu_si128((__m128i*)(y + i + 8));
            auto yy = _mm256_cvtph_ps(zz);
            auto t = _mm256_sub_ps(xx, yy);
            sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(t, t));
        }
    }
    sum1 = _mm256_add_ps(sum1, sum2);
    return reduce_add_f32x8(sum1);
#endif
}

inline float
l2a_fp16_fp16(const fp16* x, const fp16* y, const int32_t d) {
#if defined(USE_AVX512FP16) && defined(__AVX512FP16__)
    auto sum = _mm512_setzero_ph();
    for (int i = 0; i < d; i += 32) {
        auto xx = _mm512_loadu_ph(x + i);
        auto yy = _mm512_loadu_ph(y + i);
        auto d = _mm512_sub_ph(xx, yy);
        sum = _mm512_fmadd_ph(d, d, sum);
    }
    return reduce_add_f16x32(sum);
#elif defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 8) {
        auto xxx = _mm256_loadu_si256((__m256i*)(x + i));
        auto xx = _mm512_cvtph_ps(xxx);
        auto zz = _mm256_loadu_si256((__m256i*)(y + i));
        auto yy = _mm512_cvtph_ps(zz);
        auto t = _mm512_sub_ps(xx, yy);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(t, t));
    }
    return reduce_add_f32x16(sum);
#else
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 8) {
        auto xxx = _mm_loadu_si128((__m128i*)(x + i));
        auto xx = _mm256_cvtph_ps(xxx);
        auto zz = _mm_loadu_si128((__m128i*)(y + i));
        auto yy = _mm256_cvtph_ps(zz);
        auto t = _mm256_sub_ps(xx, yy);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(t, t));
    }
    return reduce_add_f32x8(sum);
#endif
}

inline float
l2a_fp32_bf16(const float* x, const bf16* y, const int32_t d) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        auto xx = _mm512_loadu_ps(x + i);
        auto zz = _mm256_loadu_si256((__m256i*)(y + i));
        auto yy = _mm512_cvtepu16_epi32(zz);
        yy = _mm512_slli_epi32(yy, 16);
        auto t = _mm512_sub_ps(xx, (__m512)yy);
        sum = _mm512_fmadd_ps(t, t, sum);
    }
    return reduce_add_f32x16(sum);
#else
    __m256 sum1 = _mm256_setzero_ps(), sum2 = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        {
            auto xx = _mm256_loadu_ps(x + i);
            auto zz = _mm_loadu_si128((__m128i*)(y + i));
            auto yy = _mm256_cvtepu16_epi32(zz);
            yy = _mm256_slli_epi32(yy, 16);
            auto t = _mm256_sub_ps(xx, (__m256)yy);
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(t, t));
        }
        {
            auto xx = _mm256_loadu_ps(x + i + 8);
            auto zz = _mm_loadu_si128((__m128i*)(y + i + 8));
            auto yy = _mm256_cvtepu16_epi32(zz);
            yy = _mm256_slli_epi32(yy, 16);
            auto t = _mm256_sub_ps(xx, (__m256)yy);
            sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(t, t));
        }
    }
    sum1 = _mm256_add_ps(sum1, sum2);
    return reduce_add_f32x8(sum1);
#endif
}

inline float
l2a_bf16_bf16(const bf16* x, const bf16* y, const int32_t d) {
#if defined(USE_AVX512BF16) && defined(__AVX512BF16__)
    auto s1 = _mm512_setzero_ps(), s2 = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 32) {
        auto xx = (__m512bh)_mm512_loadu_si512(x + i);
        auto yy = (__m512bh)_mm512_loadu_si512(y + i);
        s1 = _mm512_dpbf16_ps(s1, yy, yy);
        s2 = _mm512_dpbf16_ps(s2, xx, yy);
    }
    return reduce_add_f32x16(s1) - 2.0f * reduce_add_f32x16(s2);
#elif defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < d; i += 16) {
        auto xxx = _mm256_loadu_si256((__m256i*)(x + i));
        auto xx = _mm512_cvtepu16_epi32(xxx);
        xx = _mm512_slli_epi32(xx, 16);
        auto zz = _mm256_loadu_si256((__m256i*)(y + i));
        auto yy = _mm512_cvtepu16_epi32(zz);
        yy = _mm512_slli_epi32(yy, 16);
        auto t = _mm512_sub_ps((__m512)xx, (__m512)yy);
        sum = _mm512_fmadd_ps(t, t, sum);
    }
    return reduce_add_f32x16(sum);
#else
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < d; i += 8) {
        auto xxx = _mm_loadu_si128((__m128i*)(x + i));
        auto xx = _mm256_cvtepu16_epi32(xxx);
        xx = _mm256_slli_epi32(xx, 16);
        auto zz = _mm_loadu_si128((__m128i*)(y + i));
        auto yy = _mm256_cvtepu16_epi32(zz);
        yy = _mm256_slli_epi32(yy, 16);
        auto t = _mm256_sub_ps((__m256)xx, (__m256)yy);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(t, t));
    }
    return reduce_add_f32x8(sum);
#endif
}

inline int32_t
l2a_s8_s8(const int8_t* x, const int8_t* y, const int32_t d) {
    int32_t sum = 0;
    for (int i = 0; i < d; ++i) {
        int32_t t = int32_t(x[i]) - int32_t(y[i]);
        sum += t * t;
    }
    return sum;
}

inline int32_t
l2a_u7_u7(const uint8_t* x, const uint8_t* y, const int32_t d) {
#if defined(__AVX512VNNI__)
    __m512i sum = _mm512_setzero_epi32();
    for (int i = 0; i < d; i += 64) {
        auto xx = _mm512_loadu_si512(x + i);
        auto yy = _mm512_loadu_si512(y + i);
        auto t = _mm512_sub_epi8(xx, yy);
        t = _mm512_abs_epi8(t);
        // GCC bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=94663
        // sum = _mm512_dpbusd_epi32(sum, t, t);
        asm("vpdpbusd %1, %2, %0" : "+x"(sum) : "mx"(t), "x"(t));
    }
    return reduce_add_i32x16(sum);
#else
    __m256i sum = _mm256_setzero_si256();
    for (int i = 0; i < d; i += 32) {
        auto xx = _mm256_loadu_si256((__m256i*)(x + i));
        auto yy = _mm256_loadu_si256((__m256i*)(y + i));
        auto t = _mm256_sub_epi8(xx, yy);
        t = _mm256_abs_epi8(t);
        sum = dp_u8s8x32(sum, t, t);
    }
    return reduce_add_i32x8(sum);
#endif
}

inline int32_t
l2a_u4_u4(const uint8_t* x, const uint8_t* y, const int32_t d) {
#if defined(__AVX512VNNI__)
    __m512i sum1 = _mm512_setzero_epi32(), sum2 = _mm512_setzero_epi32();
    __m512i mask = _mm512_set1_epi8(0xf);
    for (int i = 0; i < d; i += 128) {
        auto xx = _mm512_loadu_si512((__m512i*)(x + i / 2));
        auto yy = _mm512_loadu_si512((__m512i*)(y + i / 2));
        auto xx1 = _mm512_and_si512(xx, mask);
        auto xx2 = _mm512_and_si512(_mm512_srli_epi16(xx, 4), mask);
        auto yy1 = _mm512_and_si512(yy, mask);
        auto yy2 = _mm512_and_si512(_mm512_srli_epi16(yy, 4), mask);
        auto d1 = _mm512_sub_epi8(xx1, yy1);
        auto d2 = _mm512_sub_epi8(xx2, yy2);
        d1 = _mm512_abs_epi8(d1);
        d2 = _mm512_abs_epi8(d2);
        // sum1 = _mm512_dpbusd_epi32(sum1, d1, d1);
        // sum2 = _mm512_dpbusd_epi32(sum2, d2, d2);
        asm("vpdpbusd %1, %2, %0" : "+x"(sum1) : "mx"(d1), "x"(d1));
        asm("vpdpbusd %1, %2, %0" : "+x"(sum2) : "mx"(d2), "x"(d2));
    }
    sum1 = _mm512_add_epi32(sum1, sum2);
    return reduce_add_i32x16(sum1);
#else
    __m256i sum1 = _mm256_setzero_si256(), sum2 = _mm256_setzero_si256();
    __m256i mask = _mm256_set1_epi8(0xf);
    for (int i = 0; i < d; i += 64) {
        auto xx = _mm256_loadu_si256((__m256i*)(x + i / 2));
        auto yy = _mm256_loadu_si256((__m256i*)(y + i / 2));
        auto xx1 = _mm256_and_si256(xx, mask);
        auto xx2 = _mm256_and_si256(_mm256_srli_epi16(xx, 4), mask);
        auto yy1 = _mm256_and_si256(yy, mask);
        auto yy2 = _mm256_and_si256(_mm256_srli_epi16(yy, 4), mask);
        auto d1 = _mm256_sub_epi8(xx1, yy1);
        auto d2 = _mm256_sub_epi8(xx2, yy2);
        d1 = _mm256_abs_epi8(d1);
        d2 = _mm256_abs_epi8(d2);
        sum1 = dp_u8s8x32(sum1, d1, d1);
        sum2 = dp_u8s8x32(sum2, d2, d2);
    }
    sum1 = _mm256_add_epi32(sum1, sum2);
    return reduce_add_i32x8(sum1);
#endif
}

}  // namespace helpa

#endif
