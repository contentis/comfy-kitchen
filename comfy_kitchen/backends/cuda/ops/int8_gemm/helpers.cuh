/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace comfy::int8_gemm {
namespace {

// Match the original CUTLASS epilogue: a rounded, FTZ FP32 product, then a
// fused scale/bias operation and a single round-to-nearest BF16 conversion.
__device__ __forceinline__ __nv_bfloat16 dequantize(
    int acc, float x_scale, float w_scale, __nv_bfloat16 bias) {
    float value = __int2float_rn(acc), scaled;
    asm("mul.rn.ftz.f32 %0,%1,%2;" : "=f"(scaled) : "f"(value), "f"(x_scale));
    asm("fma.rn.ftz.f32 %0,%1,%2,%3;" : "=f"(value)
        : "f"(scaled), "f"(w_scale), "f"(__bfloat162float(bias)));
    return __float2bfloat16_rn(value);
}

// Keep explicit kernel arguments: packing them into Problem changed SM100's
// load/register schedule and regressed cache-resident GEMV by about 18%.
template<int Lanes>
__global__ void gemv(const int8_t* a, const int8_t* b, const float* x_scale,
                     const float* w_scale, const __nv_bfloat16* bias,
                     __nv_bfloat16* out, int n, int k, int scalar) {
    int tid = int(blockIdx.x) * blockDim.x + threadIdx.x;
    int row = tid / Lanes, lane = threadIdx.x % Lanes, sum = 0;
    if (row < n) {
        for (int t = lane * 16; t < k; t += Lanes * 16) {
            int4 av = *reinterpret_cast<const int4*>(a + t);
            int4 bv = *reinterpret_cast<const int4*>(b + int64_t(row) * k + t);
            sum = __dp4a(av.x, bv.x, sum);
            sum = __dp4a(av.y, bv.y, sum);
            sum = __dp4a(av.z, bv.z, sum);
            sum = __dp4a(av.w, bv.w, sum);
        }
    }
    #pragma unroll
    for (int offset = Lanes / 2; offset; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset, Lanes);
    if (row < n && lane == 0)
        out[row] = dequantize(sum, x_scale[0], w_scale[scalar ? 0 : row],
                              bias ? bias[row] : __float2bfloat16(0));
}

__global__ void tiny_k(Problem p) {
    const int row = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= p.m) return;
    const int* a = reinterpret_cast<const int*>(p.a + int64_t(row) * 12);
    const int* b = reinterpret_cast<const int*>(p.b);
    int acc = __dp4a(a[0], b[0], 0);
    acc = __dp4a(a[1], b[1], acc);
    acc = __dp4a(a[2], b[2], acc);
    p.out[row] = dequantize(acc, p.x_scale[row], p.w_scale[0],
                           p.bias ? p.bias[0] : __float2bfloat16(0));
}

#if COMFY_INT8_GEMM_ARCH == 90
__global__ void reduce_dequantize(const int32_t* parts, Problem p, int splits) {
    const int64_t i = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    const int64_t total = int64_t(p.m) * p.n;
    if (i >= total) return;
    int4 acc = make_int4(0, 0, 0, 0);
    #pragma unroll 1
    for (int s = 0; s < splits; ++s) {
        const int4 value = *reinterpret_cast<const int4*>(parts + int64_t(s) * total + i);
        acc.x += value.x; acc.y += value.y;
        acc.z += value.z; acc.w += value.w;
    }
    const float x_scale = p.x_scale[i / p.n], w_scale = p.w_scale[0];
    const uint2 bias = p.bias ? *reinterpret_cast<const uint2*>(p.bias + i % p.n)
                             : make_uint2(0, 0);
    const auto convert = [&](int value, unsigned short b) {
        return unsigned(__bfloat16_as_ushort(dequantize(value, x_scale, w_scale,
                                                       __ushort_as_bfloat16(b))));
    };
    const uint2 value = make_uint2(
        convert(acc.x, bias.x & 65535) | (convert(acc.y, bias.x >> 16) << 16),
        convert(acc.z, bias.y & 65535) | (convert(acc.w, bias.y >> 16) << 16));
    *reinterpret_cast<uint2*>(p.out + i) = value;
}
#endif

}  // namespace
}  // namespace comfy::int8_gemm
