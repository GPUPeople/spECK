#pragma once

#include <utility>
#include <CUDATools/memory.h>
/*#include "../memory_space.h"
#include "../consistent_memory.h"

namespace spECK {
	template<>
	class ConsistentMemory<MemorySpace::device> : RegisteredMemory
	{
		size_t _size;
		CU::unique_ptr _ptr;

		size_t clear() override
		{
			auto s = _size;
			reset(0);
			return s;
		}
	public:
		ConsistentMemory() : _size(0)
		{
			register_consistent_memory(this);
		}

		~ConsistentMemory()
		{
			unregister_consistent_memory(this);
		}

		operator CUdeviceptr() const noexcept { return _ptr; }

		template <typename T = void>
		T* get() const noexcept { return reinterpret_cast<T*>(_ptr.operator long long unsigned int()); }

		void increaseMemRetainData(size_t size)
		{
			CU::unique_ptr tmp_ptr = CU::allocMemory(_size + size);
			cudaMemcpy(tmp_ptr.get(), _ptr.get(), _size, cudaMemcpyDeviceToDevice);
			_ptr.reset();
			_ptr = std::move(tmp_ptr);
			_size += size;
		}

		void assure(size_t size)
		{
			if (size > _size)
			{
				_ptr.reset();
				_ptr = CU::allocMemory(size);
				_size = size;
			}
		}
		void reset(size_t size = 0)
		{
			_ptr.reset();
			_size = 0;
			assure(size);
		}
	};
}*/