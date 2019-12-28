#ifndef spECK_Limits
#define spECK_Limits
#pragma once
#include <cstdint>
#include <npp.h>

namespace spECK {

template<typename _Tp>
struct numeric_limits
{
  /** The minimum finite value, or for floating types with
denormalization, the minimum positive normalized value.  */
   __host__ __device__ static constexpr _Tp
  min() noexcept { return _Tp(); }

  /** The maximum finite value.  */
  __host__ __device__ static constexpr _Tp
  max() noexcept { return _Tp(); }
};

template<>
struct numeric_limits<uint32_t>
{
  __host__ __device__ static constexpr uint32_t
 min() noexcept { return NPP_MIN_32U; }

 __host__ __device__ static constexpr uint32_t
 max() noexcept { return NPP_MAX_32U; }
};

template<>
struct numeric_limits<int32_t>
{
  __host__ __device__ static constexpr int32_t
 min() noexcept { return NPP_MIN_32S; }

 __host__ __device__ static constexpr int32_t
 max() noexcept { return NPP_MAX_32S; }
};

template<>
struct numeric_limits<uint16_t>
{
	__host__ __device__ static constexpr uint16_t
		min() noexcept { return NPP_MIN_16U; }

	__host__ __device__ static constexpr uint16_t
		max() noexcept { return NPP_MAX_16U; }
};

template<>
struct numeric_limits<int16_t>
{
	__host__ __device__ static constexpr int16_t
		min() noexcept { return NPP_MIN_16S; }

	__host__ __device__ static constexpr int16_t
		max() noexcept { return NPP_MAX_16S; }
};

template<>
struct numeric_limits<uint8_t>
{
	__host__ __device__ static constexpr uint8_t
		min() noexcept { return NPP_MIN_8U; }

	__host__ __device__ static constexpr uint8_t
		max() noexcept { return NPP_MAX_8U; }
};

template<>
struct numeric_limits<int8_t>
{
	__host__ __device__ static constexpr int8_t
		min() noexcept { return NPP_MIN_8S; }

	__host__ __device__ static constexpr int8_t
		max() noexcept { return NPP_MAX_8S; }
};
}
#endif