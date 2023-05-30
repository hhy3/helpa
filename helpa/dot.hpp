#pragma once

#include "helpa/types.hpp"

namespace helpa {

float dotu_fp32_fp16(const float *x, const fp16 *y, const int32_t d);
float dotu_fp16_fp16(const fp16 *x, const fp16 *y, const int32_t d);
float dotu_fp32_bf16(const float *x, const bf16 *y, const int32_t d);
float dotu_bf16_bf16(const bf16 *x, const bf16 *y, const int32_t d);
float dot_fp32_fp16(const float *x, const fp16 *y, const int32_t d);
float dot_fp16_fp16(const fp16 *x, const fp16 *y, const int32_t d);
float dot_fp32_bf16(const float *x, const bf16 *y, const int32_t d);
float dot_bf16_bf16(const bf16 *x, const bf16 *y, const int32_t d);

} // namespace helpa
