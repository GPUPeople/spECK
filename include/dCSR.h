#pragma once

#include <cstddef>
#include <algorithm>

template<typename T>
struct CSR;

template<typename T>
struct dCSR
{
	size_t rows, cols, nnz;

	T* data;
	unsigned int* row_offsets;
	unsigned int* col_ids;

	dCSR() : rows(0), cols(0), nnz(0), data(nullptr), row_offsets(nullptr), col_ids(nullptr) { }
	void alloc(size_t rows, size_t cols, size_t nnz, bool allocOffsets = true);
	void reset();
	virtual ~dCSR();
};

template <typename T>
struct dCSRNoDealloc
{
	size_t rows, cols, nnz;

	T* data;
	unsigned int* row_offsets;
	unsigned int* col_ids;

	dCSRNoDealloc(const dCSR<T>& a) : rows(a.rows), cols(a.cols), data(a.data), nnz(a.nnz), row_offsets(a.row_offsets), col_ids(a.col_ids) {}
	dCSRNoDealloc() = default;
};

template<typename T>
void convert(dCSR<T>& dcsr, const CSR<T>& csr, unsigned int padding = 0);

template<typename T>
void convert(dCSR<T>& dcsr, const dCSR<T>& csr, unsigned int padding = 0);

template<typename T>
void convert(CSR<T>& csr, const dCSR<T>& dcsr, unsigned int padding = 0);

template<typename T>
void convert(CSR<T>& csr, const CSR<T>& dcsr, unsigned int padding = 0);