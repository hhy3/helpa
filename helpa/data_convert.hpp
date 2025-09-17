#pragma once

#include "helpa/common.hpp"
#include "helpa/types.hpp"

namespace helpa {

inline void
fp32_to_fp16(const float* from, fp16* to, size_t size);

inline void
fp32_to_bf16(const float* from, bf16* to, size_t size);

inline void
fp16_to_fp32(const fp16* from, float* to, size_t size);

inline void
bf16_to_fp32(const bf16* from, float* to, size_t size);

inline void
fp32_to_fp16_align(const float* from, fp16* to, size_t size);

inline void
fp32_to_bf16_align(const float* from, bf16* to, size_t size);

inline void
fp16_to_fp32_align(const fp16* from, float* to, size_t size);

inline void
bf16_to_fp32_align(const bf16* from, float* to, size_t size);

}  // namespace helpa

#if defined(HELPA_USE_X86)
#include "helpa/platform/x86/data_convert.hpp"
#else
#include "helpa/platform/scalar/data_convert.hpp"
#endif
