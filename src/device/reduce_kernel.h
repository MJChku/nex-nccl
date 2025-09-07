#ifndef NCCL_REDUCE_KERNEL_CPU_H_
#define NCCL_REDUCE_KERNEL_CPU_H_

#include <cstdint>
#include <type_traits>
#include "op128.h"

// CPU qualifiers
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__
#endif

////////////////////////////////////////////////////////////////////////////////
// Floating-point trait

template<typename T>
struct IsFloatingPoint : std::false_type {};
template<> struct IsFloatingPoint<float>  : std::true_type {};
template<> struct IsFloatingPoint<double> : std::true_type {};

////////////////////////////////////////////////////////////////////////////////
// Reduction function classes: no-op constructors

template<typename T>
struct FuncCopy { using EltType = T; __host__ __device__ FuncCopy(uint64_t=0){} };

template<typename T>
struct FuncSum { using EltType = T; __host__ __device__ FuncSum(uint64_t=0){} };

template<typename T>
struct FuncProd { using EltType = T; __host__ __device__ FuncProd(uint64_t=0){} };

template<typename T>
struct FuncMinMax {
  using EltType = T;
  uint64_t xormask;
  bool isMin;
  __host__ __device__ FuncMinMax(uint64_t op=0): xormask(op), isMin((op&1)==0) {}
};

template<typename T>
struct FuncPreMulSum { using EltType = T; __host__ __device__ FuncPreMulSum(uint64_t=0){} };

template<typename T>
struct FuncSumPostDiv { using EltType = T;
  __host__ __device__ FuncSumPostDiv(uint64_t=0){}
  __host__ __device__ T divide(T x) const { return x; }
};

////////////////////////////////////////////////////////////////////////////////
// RedOpArg trait

template<typename Fn>
struct RedOpArg { static constexpr bool ArgUsed = false;
  __host__ __device__ static uint64_t loadArg(void*) { return 0; }
};

template<typename T>
struct RedOpArg<FuncMinMax<T>> {
  static constexpr bool ArgUsed = true;
  __host__ __device__ static uint64_t loadArg(void* ptr) {
    return *reinterpret_cast<uint64_t*>(ptr);
  }
};

template<typename T>
struct RedOpArg<FuncPreMulSum<T>> : RedOpArg<FuncMinMax<T>> {};

template<typename T>
struct RedOpArg<FuncSumPostDiv<T>> {
  static constexpr bool ArgUsed = true;
  __host__ __device__ static uint64_t loadArg(void* ptr) {
    return *reinterpret_cast<uint64_t*>(ptr);
  }
};

////////////////////////////////////////////////////////////////////////////////
// Packing and reduction trait stubs

template<typename A, typename B, int EltPerPack>
struct Apply_Cast {
  __host__ __device__ static BytePack<EltPerPack*sizeof(B)> cast(BytePack<EltPerPack*sizeof(A)>) {
    return BytePack<EltPerPack*sizeof(B)>();
  }
};

template<typename Fn, int EltPerPack>
struct Apply_Reduce {
  __host__ __device__ static BytePack<EltPerPack*sizeof(typename Fn::EltType)>
  reduce(Fn, BytePack<EltPerPack*sizeof(typename Fn::EltType)>, BytePack<EltPerPack*sizeof(typename Fn::EltType)>) {
    return BytePack<EltPerPack*sizeof(typename Fn::EltType)>();
  }
};

template<typename Fn, int EltPerPack>
struct Apply_PreOp {
  static constexpr bool IsIdentity = true;
  __host__ __device__ static BytePack<EltPerPack*sizeof(typename Fn::EltType)> preOp(Fn, BytePack<EltPerPack*sizeof(typename Fn::EltType)>) {
    return BytePack<EltPerPack*sizeof(typename Fn::EltType)>();
  }
};

template<typename Fn, int EltPerPack>
struct Apply_PostOp {
  static constexpr bool IsIdentity = true;
  __host__ __device__ static BytePack<EltPerPack*sizeof(typename Fn::EltType)> postOp(Fn, BytePack<EltPerPack*sizeof(typename Fn::EltType)>) {
    return BytePack<EltPerPack*sizeof(typename Fn::EltType)>();
  }
};

template<typename Fn, int BytePerPack>
struct Apply_LoadMultimem {
  __host__ __device__ static BytePack<BytePerPack> load(Fn, uintptr_t) {
    return BytePack<BytePerPack>();
  }
};

template<typename Fn>
struct LoadMultimem_BigPackSize { static constexpr int BigPackSize = 0; };

////////////////////////////////////////////////////////////////////////////////
// Public API stubs

template<typename A, typename B, typename PackA>
__host__ __device__ BytePack<BytePackOf<PackA>::Size*sizeof(B)/sizeof(A)>
applyCast(PackA) {
  return BytePack<BytePackOf<PackA>::Size*sizeof(B)/sizeof(A)>();
}

template<typename Fn, typename Pack>
__host__ __device__ Pack applyReduce(Fn, Pack, Pack) { return Pack(); }

template<typename Fn, typename Pack>
__host__ __device__ Pack applyPreOp(Fn, Pack) { return Pack(); }

template<typename Fn, typename Pack>
__host__ __device__ Pack applyPostOp(Fn, Pack) { return Pack(); }

template<typename Fn, int BytePerPack>
__host__ __device__ BytePack<BytePerPack> applyLoadMultimem(Fn, uintptr_t) {
  return BytePack<BytePerPack>();
}

#endif // NCCL_REDUCE_KERNEL_CPU_H_
