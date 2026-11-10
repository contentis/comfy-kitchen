/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cublas_v2.h>
#include <map>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace comfy::int8_gemm {

// Only the two measured Hopper split-K routes need cuBLAS (not cuBLASLt).
// Load it on demand, like the existing cuBLASLt backend, without a new wheel
// dependency. Handles belong to a host thread and device; each launch assigns
// its stream and private workspace.
struct Cublas {
    decltype(&cublasCreate) create = nullptr;
    decltype(&cublasDestroy) destroy = nullptr;
    decltype(&cublasSetStream) set_stream = nullptr;
    decltype(&cublasSetWorkspace) set_workspace = nullptr;
    using StridedGemm = cublasStatus_t (CUBLASWINAPI *)(
        cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
        const void*, const void*, cudaDataType, int, long long,
        const void*, cudaDataType, int, long long, const void*, void*, cudaDataType,
        int, long long, int, cublasComputeType_t, cublasGemmAlgo_t);
    StridedGemm gemm = nullptr;

    Cublas() {
#ifdef _WIN32
        HMODULE library = LoadLibraryA("cublas64_13.dll");
        if (!library) library = LoadLibraryA("cublas64_12.dll");
        const auto symbol = [&](const char* name) { return GetProcAddress(library, name); };
#else
        void* library = dlopen("libcublas.so.13", RTLD_NOW | RTLD_LOCAL);
        if (!library) library = dlopen("libcublas.so.12", RTLD_NOW | RTLD_LOCAL);
        const auto symbol = [&](const char* name) { return dlsym(library, name); };
#endif
        if (!library) return;
        create = reinterpret_cast<decltype(create)>(symbol("cublasCreate_v2"));
        destroy = reinterpret_cast<decltype(destroy)>(symbol("cublasDestroy_v2"));
        set_stream = reinterpret_cast<decltype(set_stream)>(symbol("cublasSetStream_v2"));
        set_workspace = reinterpret_cast<decltype(set_workspace)>(symbol("cublasSetWorkspace_v2"));
        gemm = reinterpret_cast<decltype(gemm)>(symbol("cublasGemmStridedBatchedEx"));
        // Keep the library loaded while cached handles and captured graphs exist.
    }

    cublasHandle_t handle(cudaStream_t stream) const {
        if (!create || !destroy || !set_stream || !set_workspace || !gemm) return nullptr;
        int device;
        if (cudaGetDevice(&device) != cudaSuccess) return nullptr;
        struct Handles {
            const Cublas* api;
            std::map<int, cublasHandle_t> values;
            ~Handles() { for (const auto& entry : values) api->destroy(entry.second); }
        };
        thread_local Handles handles{this, {}};
        auto found = handles.values.find(device);
        if (found != handles.values.end()) return found->second;
        cudaStreamCaptureStatus capture;
        if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
            capture != cudaStreamCaptureStatusNone) return nullptr;
        cublasHandle_t result;
        if (create(&result) != CUBLAS_STATUS_SUCCESS) return nullptr;
        handles.values.emplace(device, result);
        return result;
    }
};

inline bool split_gemm(const Problem& p, int splits, cudaStream_t stream) {
    static const Cublas api;
    auto handle = api.handle(stream);
    if (!handle) return false;
    // One allocation owns both cuBLAS scratch and integer partials. Graph
    // capture owns its allocation/free nodes; concurrent graphs cannot alias it.
    constexpr size_t bytes = 32u << 20;
    const size_t partial_bytes = size_t(p.m) * p.n * splits * sizeof(int32_t);
    void* workspace = nullptr;
    if (cudaMallocAsync(&workspace, bytes + partial_bytes, stream) != cudaSuccess)
        throw std::runtime_error("INT8 split-K workspace allocation failed");
    auto* partials = reinterpret_cast<int32_t*>(static_cast<char*>(workspace) + bytes);
    if (api.set_stream(handle, stream) != CUBLAS_STATUS_SUCCESS ||
        api.set_workspace(handle, workspace, bytes) != CUBLAS_STATUS_SUCCESS) {
        cudaFreeAsync(workspace, stream);
        throw std::runtime_error("INT8 split-K cuBLAS setup failed");
    }
    const int alpha = 1, beta = 0;
    auto status = api.gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, p.n, p.m, p.k / splits,
        &alpha, p.b, CUDA_R_8I, p.k, p.k / splits, p.a, CUDA_R_8I, p.k, p.k / splits,
        &beta, partials, CUDA_R_32I, p.n, int64_t(p.m) * p.n, splits,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    if (status == CUBLAS_STATUS_SUCCESS)
        reduce_dequantize<<<(int64_t(p.m) * p.n + 1023) / 1024, 256, 0, stream>>>(partials, p, splits);
    const auto freed = cudaFreeAsync(workspace, stream);
    if (status != CUBLAS_STATUS_SUCCESS || freed != cudaSuccess)
        throw std::runtime_error("INT8 split-K GEMM failed");
    return true;
}

}  // namespace comfy::int8_gemm
