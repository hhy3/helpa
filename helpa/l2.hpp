#pragma once

#include <cstdint>

#include "helpa/types.hpp"

namespace helpa {

float l2u_fp32_fp32(const float *x, const float *y, const int32_t d);
float l2u_fp32_fp16(const float *x, const fp16 *y, const int32_t d);
float l2u_fp16_fp16(const fp16 *x, const fp16 *y, const int32_t d);
float l2u_fp32_bf16(const float *x, const bf16 *y, const int32_t d);
float l2u_bf16_bf16(const bf16 *x, const bf16 *y, const int32_t d);
float l2_fp32_fp32(const float *x, const float *y, const int32_t d);
float l2_fp32_fp16(const float *x, const fp16 *y, const int32_t d);
float l2_fp16_fp16(const fp16 *x, const fp16 *y, const int32_t d);
float l2_fp32_bf16(const float *x, const bf16 *y, const int32_t d);
float l2_bf16_bf16(const bf16 *x, const bf16 *y, const int32_t d);

} // namespace helpa
