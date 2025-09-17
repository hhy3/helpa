#pragma once

#include "helpa/types.hpp"

namespace helpa {

inline void
fp32_to_fp16_ref(const float* from, fp16* to, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        to[i] = fp16(from[i]);
    }
}

inline void
fp32_to_bf16_ref(const float* from, bf16* to, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        to[i] = bf16(from[i]);
    }
}

inline void
fp16_to_fp32_ref(const fp16* from, float* to, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        to[i] = float(from[i]);
    }
}

inline void
bf16_to_fp32_ref(const bf16* from, float* to, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        to[i] = float(from[i]);
    }
}

}  // namespace helpa