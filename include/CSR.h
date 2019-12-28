#pragma once

#include <memory>
#include <algorithm>
#include <math.h>
#include <cstring>

template<typename T>
struct COO;

template<typename T>
struct DenseVector;

template<typename T>
struct CSR
{
	struct Statistics
	{
		double mean;
		double std_dev;
		size_t max;
		size_t min;
	};

	void computeStatistics(double& mean, double& std_dev, size_t& max, size_t& min)
	{
		// running variance by Welford
		size_t count = 0;
		mean = 0;
		double M2 = 0;
		max = 0;
		min = cols;
		for (size_t i = 0; i < rows; ++i)
		{
			size_t r_length = row_offsets[i + 1] - row_offsets[i];
			min = std::min(min, r_length);
			max = std::max(max, r_length);
			++count;
			double newValue = static_cast<double>(r_length);
			double delta = newValue - mean;
			mean = mean + delta / count;
			double delta2 = newValue - mean;
			M2 = M2 + delta * delta2;
		}
		if (count < 2)
			std_dev = 0;
		else
			std_dev = sqrt(M2 / (count - 1));
	}

	Statistics rowStatistics()
	{
		Statistics stats;
		computeStatistics(stats.mean, stats.std_dev, stats.max, stats.min);
		return stats;
	}

	size_t rows, cols, nnz;

	std::unique_ptr<T[]> data;
	std::unique_ptr<unsigned int[]> row_offsets;
	std::unique_ptr<unsigned int[]> col_ids;

	CSR() : rows(0), cols(0), nnz(0) { }
	void alloc(size_t rows, size_t cols, size_t nnz);

	// CSR<T>& operator=(CSR<T> other)
	// {
	// 	this->rows = other.rows;
	// 	this->cols = other.cols;
	// 	this->nnz = other.nnz;
	// 	this->data = std::move(other.data);
	// 	this->row_offsets = std::move(other.row_offsets);
	// 	this->col_ids = std::move(other.col_ids);
	// 	return *this;
	// }

	// CSR(const CSR<T>& other)
	// {
	// 	this->rows = other.rows;
	// 	this->cols = other.cols;
	// 	this->nnz = other.nnz;
	// 	this->data = std::make_unique<T[]>(other.nnz);
	// 	memcpy(this->data.get(), other.data.get(), sizeof(T) * other.nnz);
	// 	this->col_ids = std::make_unique<unsigned int[]>(other.nnz);
	// 	memcpy(this->col_ids.get(), other.col_ids.get(), sizeof(unsigned int) * other.nnz);
	// 	this->row_offsets = std::make_unique<unsigned int[]>(other.rows + 1);
	// 	memcpy(this->row_offsets.get(), other.row_offsets.get(), sizeof(unsigned int) * (other.rows + 1));
	// }

};


template<typename T>
CSR<T> loadCSR(const char* file);
template<typename T>
void storeCSR(const CSR<T>& mat, const char* file);

template<typename T>
void spmv(DenseVector<T>& res, const CSR<T>& m, const DenseVector<T>& v, bool transpose = false);

template<typename T>
void convert(CSR<T>& res, const COO<T>& coo);
