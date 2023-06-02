#pragma once

#include "helpa/types.hpp"

namespace helpa {

inline float dot_fp32_fp32(const float *x, const float *y, const int32_t d);
inline float dot_fp32_fp16(const float *x, const fp16 *y, const int32_t d);
inline float dot_fp16_fp16(const fp16 *x, const fp16 *y, const int32_t d);
inline float dot_fp32_bf16(const float *x, const bf16 *y, const int32_t d);
inline float dot_bf16_bf16(const bf16 *x, const bf16 *y, const int32_t d);
inline int32_t dot_s8_s8(const int8_t *x, const int8_t *y, const int32_t d);
inline int32_t dot_u8_u8(const int8_t *x, const int8_t *y, const int32_t d);
inline float dotu_fp32_fp32(const float *x, const float *y, const int32_t d);
inline float dotu_fp32_fp16(const float *x, const fp16 *y, const int32_t d);
inline float dotu_fp16_fp16(const fp16 *x, const fp16 *y, const int32_t d);
inline float dotu_fp32_bf16(const float *x, const bf16 *y, const int32_t d);
inline float dotu_bf16_bf16(const bf16 *x, const bf16 *y, const int32_t d);
inline int32_t dotu_s8_s8(const int8_t *x, const int8_t *y, const int32_t d);
inline int32_t dotu_u8_u8(const uint8_t *x, const uint8_t *y, const int32_t d);

} // namespace helpa

#if defined(__AVX512F__)
#include "helpa/platform/avx512/dot_impl.hpp"
#elif defined(__AVX2__)
#include "helpa/platform/avx2/dot_impl.hpp"
#elif defined(__aarch64__)
#include "helpa/platform/neon/dot_impl.hpp"
#else
#include "helpa/platform/scalar/dot_impl.hpp"
#endif
