#pragma once

#include "Vector.h"

#include <memory>


template<typename T>
struct COO
{
	size_t rows, cols, nnz;

	std::unique_ptr<T[]> data;
	std::unique_ptr<unsigned int[]> row_ids;
	std::unique_ptr<unsigned int[]> col_ids;

	COO() : rows(0), cols(0), nnz(0) { }
	void alloc(size_t rows, size_t cols, size_t nnz);
};

template<typename T>
COO<T> loadMTX(const char* file);
template<typename T>
COO<T> loadCOO(const char* file);
template<typename T>
void storeCOO(const COO<T>& mat, const char* file);

template<typename T>
void spmv(DenseVector<T>& res, const COO<T>& m, const DenseVector<T>& v, bool transpose = false);
