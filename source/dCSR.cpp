#include "dCSR.h"
#include "CSR.h"

#include <cuda_runtime.h>

namespace
{
	template<typename T>
	void dealloc(dCSR<T>& mat)
	{
		if (mat.col_ids != nullptr)
			cudaFree(mat.col_ids);
		if (mat.data != nullptr)
			cudaFree(mat.data);
		if (mat.row_offsets != nullptr)
			cudaFree(mat.row_offsets);
		mat.col_ids = nullptr;
		mat.data = nullptr;
		mat.row_offsets = nullptr;
		mat.nnz = 0;
		mat.rows = 0;
	}
}

template<typename T>
void dCSR<T>::alloc(size_t r, size_t c, size_t n, bool allocOffsets)
{
	dealloc(*this);
	rows = r;
	cols = c;
	nnz = n;
	cudaMalloc(&data, sizeof(T)*n);
	cudaMalloc(&col_ids, sizeof(unsigned int)*n);
	if (allocOffsets)
		cudaMalloc(&row_offsets, sizeof(unsigned int)*(r+1));
}
template<typename T>
dCSR<T>::~dCSR()
{
	dealloc(*this);
}

template<typename T>
void dCSR<T>::reset()
{
	dealloc(*this);
}


template<typename T>
void convert(dCSR<T>& dst, const CSR<T>& src, unsigned int padding)
{
	dst.alloc(src.rows + padding, src.cols, src.nnz + 8*padding);
	dst.rows = src.rows; dst.nnz = src.nnz; dst.cols = src.cols;
	cudaMemcpy(dst.data, &src.data[0], src.nnz * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(dst.col_ids, &src.col_ids[0], src.nnz * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(dst.row_offsets, &src.row_offsets[0], (src.rows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);

	if (padding)
	{
		cudaMemset(dst.data + src.nnz, 0, 8 * padding * sizeof(T));
		cudaMemset(dst.col_ids + src.nnz, 0, 8 * padding * sizeof(unsigned int));
		cudaMemset(dst.row_offsets + src.rows + 1, 0, padding * sizeof(unsigned int));
	}
}

template<typename T>
void convert(CSR<T>& dst, const dCSR<T>& src, unsigned int padding)
{
	dst.alloc(src.rows + padding, src.cols, src.nnz + 8 * padding);
	dst.rows = src.rows; dst.nnz = src.nnz; dst.cols = src.cols;
	cudaMemcpy(dst.data.get(), src.data, dst.nnz * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(dst.col_ids.get(), src.col_ids, dst.nnz * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(dst.row_offsets.get(), src.row_offsets, (dst.rows + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
}

template<typename T>
void convert(dCSR<T>& dst, const dCSR<T>& src, unsigned int padding)
{
	dst.alloc(src.rows + padding, src.cols, src.nnz + 8 * padding);
	dst.rows = src.rows; dst.nnz = src.nnz; dst.cols = src.cols;
	cudaMemcpy(dst.data, src.data, dst.nnz * sizeof(T), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dst.col_ids, src.col_ids, dst.nnz * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dst.row_offsets, src.row_offsets, (dst.rows + 1) * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
}

template<typename T>
void convert(CSR<T>& dst, const CSR<T>& src, unsigned int padding)
{
	dst.alloc(src.rows + padding, src.cols, src.nnz + 8 * padding);
	dst.rows = src.rows; dst.nnz = src.nnz; dst.cols = src.cols;
	memcpy(dst.data.get(), src.data.get(), dst.nnz * sizeof(T));
	memcpy(dst.col_ids.get(), src.col_ids.get(), dst.nnz * sizeof(unsigned int));
	memcpy(dst.row_offsets.get(), src.row_offsets.get(), (dst.rows + 1) * sizeof(unsigned int));
}

template void dCSR<float>::alloc(size_t r, size_t c, size_t n, bool allocOffsets);
template void dCSR<double>::alloc(size_t r, size_t c, size_t n, bool allocOffsets);
// template void dCSR<uint64_t>::alloc(size_t r, size_t c, size_t n, bool allocOffsets);

template dCSR<float>::~dCSR();
template dCSR<double>::~dCSR();
// template dCSR<uint64_t>::~dCSR();

template void dCSR<float>::reset();
template void dCSR<double>::reset();
// template void dCSR<uint64_t>::reset();

template void convert(dCSR<float>& dcsr, const CSR<float>& csr, unsigned int);
template void convert(dCSR<double>& dcsr, const CSR<double>& csr, unsigned int);
// template void convert(dCSR<uint64_t>& dcsr, const CSR<uint64_t>& csr, unsigned int);

template void convert(CSR<float>& csr, const dCSR<float>& dcsr, unsigned int padding);
template void convert(CSR<double>& csr, const dCSR<double>& dcsr, unsigned int padding);
// template void convert(CSR<uint64_t>& csr, const dCSR<uint64_t>& dcsr, unsigned int padding);

template void convert(dCSR<float>& dcsr, const dCSR<float>& csr, unsigned int);
template void convert(dCSR<double>& dcsr, const dCSR<double>& csr, unsigned int);
// template void convert(dCSR<uint64_t>& dcsr, const dCSR<uint64_t>& csr, unsigned int);

template void convert(CSR<float>& csr, const CSR<float>& dcsr, unsigned int padding);
template void convert(CSR<double>& csr, const CSR<double>& dcsr, unsigned int padding);
// template void convert(CSR<uint64_t>& csr, const CSR<uint64_t>& dcsr, unsigned int padding);