/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "int8_gemm.cuh"
#include "routes.cuh"
#include <stdexcept>
#include <cstdlib>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

#if COMFY_INT8_GEMM_ARCH == 90
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "sm90_narrow.cuh"
#else
#include "cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp"
#endif
#include "helpers.cuh"
#if COMFY_INT8_GEMM_ARCH == 90
#include "cublas.cuh"
#include "sm80.cuh"
#endif

namespace comfy::int8_gemm {
namespace {
using namespace cute;
using Out = cutlass::bfloat16_t;
namespace fusion = cutlass::epilogue::fusion;

// One DQ recipe and launch body serve every architecture/tile specialization.
template<int M, int N, int CM, int CN, int Stages = 0, int EpiN = 0>
struct NativeGemm {
    using Tile = Shape<Int<M>, Int<N>, _128>;
    using Cluster = Shape<Int<CM>, Int<CN>, _1>;
    static constexpr auto Round = cutlass::FloatRoundStyle::round_to_nearest;
    using XScale = fusion::Sm90ColBroadcast<0, Tile, float, float, Stride<_1,_0,_0>, 4, false>;
    using WScale = fusion::Sm90ScalarBroadcast<float>;
    using Bias = fusion::Sm90RowBroadcast<0, Tile, Out, float, Stride<_0,_1,_0>, 8, true>;
    using Mul = fusion::Sm90Compute<cutlass::multiplies, float, float, Round>;
    using Add = fusion::Sm90Compute<cutlass::plus, Out, float, Round>;
    using First = fusion::Sm90EVT<Mul, fusion::Sm90AccFetch, XScale>;
    using Second = fusion::Sm90EVT<Mul, First, WScale>;
    using EVT = fusion::Sm90EVT<Add, Second, Bias>;
#if COMFY_INT8_GEMM_ARCH == 90
    using Arch = cutlass::arch::Sm90;
    using EpiSchedule = std::conditional_t<Stages != 0, cutlass::epilogue::TmaWarpSpecialized,
                                          cutlass::epilogue::TmaWarpSpecializedCooperative>;
    using MainSchedule = std::conditional_t<Stages != 0, cutlass::gemm::KernelTmaWarpSpecialized,
                                           cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
    using EpiTile = cutlass::epilogue::collective::EpilogueTileAuto;
#else
    using Arch = cutlass::arch::Sm100;
    using EpiSchedule = std::conditional_t<CM == 2, cutlass::epilogue::TmaWarpSpecialized2Sm,
                                          cutlass::epilogue::TmaWarpSpecialized1Sm>;
    using MainSchedule = std::conditional_t<CM == 2, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
                                           cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>;
    using EpiTile = std::conditional_t<EpiN == 0, cutlass::epilogue::collective::EpilogueTileAuto,
                                      Shape<_128, Int<EpiN>>>;
#endif
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        Arch, cutlass::arch::OpClassTensorOp, Tile, Cluster, EpiTile, int32_t, float,
        void, cutlass::layout::RowMajor, 8, Out, cutlass::layout::RowMajor, 8,
        EpiSchedule, EVT>::CollectiveOp;
    using StageCount = std::conditional_t<Stages != 0, cutlass::gemm::collective::StageCount<Stages>,
        cutlass::gemm::collective::StageCountAutoCarveout<int(sizeof(typename Epilogue::SharedStorage))>>;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch, cutlass::arch::OpClassTensorOp,
        int8_t, cutlass::layout::RowMajor, 16, int8_t, cutlass::layout::ColumnMajor, 16,
        int32_t, Tile, Cluster, StageCount, MainSchedule>::CollectiveOp;
    using ProblemShape = cute::Shape<int,int,int,int>;
    using Scheduler = cutlass::gemm::PersistentScheduler;
#if COMFY_INT8_GEMM_ARCH == 90
    using Kernel = std::conditional_t<Stages != 0,
        cutlass::gemm::kernel::NarrowGemm<ProblemShape, Mainloop, Epilogue, Scheduler>,
        cutlass::gemm::kernel::GemmUniversal<ProblemShape, Mainloop, Epilogue, Scheduler>>;
#else
    using Kernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, Mainloop, Epilogue, Scheduler>;
#endif
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    static bool run(const Problem& p, int swizzle, cudaStream_t stream) {
        const auto a = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, make_shape(p.m,p.k,1));
        const auto b = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, make_shape(p.n,p.k,1));
        const auto c = cutlass::make_cute_packed_stride(typename Kernel::StrideC{}, make_shape(p.m,p.n,1));
        const auto d = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, make_shape(p.m,p.n,1));
        typename WScale::Arguments weight{};
        weight.scalar_ptrs[0] = p.w_scale;
        typename EVT::Arguments callback = {
            {{{}, {p.x_scale, 0.f, {}}, {}}, weight, {}},
            {reinterpret_cast<const Out*>(p.bias), Out(0), {}}, {}};
        cutlass::KernelHardwareInfo hardware;
        if (cudaGetDevice(&hardware.device_id) != cudaSuccess ||
            cudaDeviceGetAttribute(&hardware.sm_count, cudaDevAttrMultiProcessorCount,
                                   hardware.device_id) != cudaSuccess) return false;
        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemm, {p.m,p.n,p.k,1},
            {p.a,a,p.b,b}, {callback,nullptr,c,reinterpret_cast<Out*>(p.out),d}, hardware};
        using Raster = decltype(args.scheduler.raster_order);
        args.scheduler.raster_order = swizzle < 0 ? Raster::AlongN : Raster::AlongM;
        args.scheduler.max_swizzle_size = std::abs(swizzle);
        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;
        const size_t bytes = Gemm::get_workspace_size(args);
        void* workspace = nullptr;
        if (bytes && cudaMallocAsync(&workspace, bytes, stream) != cudaSuccess)
            throw std::runtime_error("INT8 TMA workspace allocation failed");
        auto status = gemm.initialize(args, workspace, stream);
        if (status == cutlass::Status::kSuccess) status = gemm.run(stream);
        const auto freed = workspace ? cudaFreeAsync(workspace, stream) : cudaSuccess;
        if (status != cutlass::Status::kSuccess || freed != cudaSuccess)
            throw std::runtime_error("INT8 TMA GEMM failed");
        return true;
    }
};

bool launch(const Problem& p, cudaStream_t stream) {
    if (p.n == 1 && p.k == 12) {
        tiny_k<<<(p.m + 255) / 256, 256, 0, stream>>>(p);
        return true;
    }
    if (p.m == 1 && p.k % 16 == 0) {
#if COMFY_INT8_GEMM_ARCH == 100
        constexpr int Block = 128;
        const bool wide = (p.n == 256 && p.k == 1024) || (p.n == 18432 && p.k == 3072);
        if (wide) {
            gemv<32><<<(p.n * 32 + Block - 1) / Block, Block, 0, stream>>>(
                p.a, p.b, p.x_scale, p.w_scale, p.bias, p.out, p.n, p.k, 1);
            return true;
        }
#else
        constexpr int Block = 256;
#endif
        gemv<16><<<(p.n * 16 + Block - 1) / Block, Block, 0, stream>>>(
                p.a, p.b, p.x_scale, p.w_scale, p.bias, p.out, p.n, p.k, 1);
        return true;
    }
    const Route* route = find_route(p.m, p.n, p.k, p.bias != nullptr);
    if (!route) return false;
#if COMFY_INT8_GEMM_ARCH == 90
    const int swizzle = route->hopper_swizzle;
    switch (route->hopper) {
        case -3:
            return FusedInt8Gemm<Out,64,128,128,32,64,128,3,cutlass::arch::Sm80,true>::run_strided(
                p.a, p.b, p.x_scale, p.w_scale, reinterpret_cast<const Out*>(p.bias),
                reinterpret_cast<Out*>(p.out), p.m, p.n, p.k, p.n, stream);
        case -2: case -1: return split_gemm(p, -route->hopper, stream);
        case 0: return NativeGemm<64,128,1,1,4>::run(p, swizzle, stream);
        case 1: return NativeGemm<64,128,1,1,6>::run(p, swizzle, stream);
        case 2: return NativeGemm<128,256,2,1>::run(p, swizzle, stream);
        case 3: return NativeGemm<256,192,1,2>::run(p, swizzle, stream);
    }
#else
    const int swizzle = route->blackwell_swizzle;
    switch (route->blackwell) {
        case 0: return NativeGemm<64,64,1,1>::run(p, swizzle, stream);
        case 1: return NativeGemm<64,128,1,1>::run(p, swizzle, stream);
        case 2: return NativeGemm<128,128,1,1>::run(p, swizzle, stream);
        case 3: return NativeGemm<256,64,2,1>::run(p, swizzle, stream);
        case 4: return NativeGemm<256,192,2,1>::run(p, swizzle, stream);
        case 5: return NativeGemm<256,256,2,1>::run(p, swizzle, stream);
        case 6: return NativeGemm<256,256,2,1,0,64>::run(p, swizzle, stream);
    }
#endif
    return false;
}
}  // namespace

#if COMFY_INT8_GEMM_ARCH == 90
bool launch_sm90(const Problem& p, cudaStream_t stream) {
#else
bool launch_sm100(const Problem& p, cudaStream_t stream) {
#endif
    const bool launched = launch(p, stream);
    if (launched && cudaGetLastError() != cudaSuccess)
        throw std::runtime_error("INT8 GEMM launch failed");
    return launched;
}

}  // namespace comfy::int8_gemm
