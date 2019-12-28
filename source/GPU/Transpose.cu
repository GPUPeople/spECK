// Global includes
#include <thrust/device_vector.h>
#include <stdint.h>
#include "device_launch_parameters.h"

// Local includes
#include "Transpose.h"
#include "common.h"

__global__ void d_calulateTransposeDistribution(int in_rows, int in_cols,
	const uint32_t* __restrict input_offset, const uint32_t* __restrict input_indices, uint32_t* output_offset)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= in_rows)
		return;

	uint32_t offset = input_offset[tid];
	uint32_t number_entries = input_offset[tid + 1] - offset;

	for (uint32_t i = 0; i < number_entries; ++i)
	{
		atomicAdd(output_offset + input_indices[offset + i], 1);
	}

	return;
}

template <typename DataType>
__global__ void d_findPosition(int in_rows, int in_cols, const uint32_t* __restrict input_offset, const uint32_t* __restrict input_indices,
	const DataType* __restrict input_values, uint32_t* output_offset, uint32_t* output_indices, DataType* output_values, uint32_t* helper, uint32_t* helper_position)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= in_rows)
		return;

	uint32_t offset = input_offset[tid];
	uint32_t number_entries = input_offset[tid + 1] - offset;

	for (uint32_t i = 0; i < number_entries; ++i)
	{
		uint32_t row_index = input_indices[offset + i];
		uint32_t insert_position = atomicAdd(helper + row_index, 1);
		uint32_t o_offset = output_offset[row_index];
		helper_position[o_offset + insert_position] = tid;
	}

	return;
}

template <typename DataType>
__global__ void d_writeTranspose(int in_rows, int in_cols, const uint32_t* __restrict input_offset, const uint32_t* __restrict input_indices,
	const DataType* __restrict input_values, uint32_t* output_offset, uint32_t* output_indices, DataType* output_values, uint32_t* helper, uint32_t* helper_position)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= in_rows)
		return;

	uint32_t offset = input_offset[tid];
	uint32_t number_entries = input_offset[tid + 1] - offset;

	for (uint32_t i = 0; i < number_entries; ++i)
	{
		uint32_t row_index = input_indices[offset + i];
		uint32_t actual_position(0);
		uint32_t entries_output = helper[row_index];
		uint32_t o_offset = output_offset[row_index];
		for (uint32_t j = 0; j < entries_output; ++j)
		{
			if (helper_position[o_offset + j] < tid)
				++actual_position;
		}		
		output_indices[o_offset + actual_position] = tid;
		output_values[o_offset + actual_position] = input_values[offset + i];
	}

	return;
}


namespace spECK {
	template <typename DataType>
	void Transpose(const dCSR<DataType>& matIn, dCSR<DataType>& matTransposeOut)
	{
		int blockSize(256);
		int gridSize(divup<int>(matIn.rows + 1, blockSize));

		matTransposeOut.alloc(matIn.cols, matIn.rows, matIn.nnz);

		// Allocate and set helper resources, Memset output vector
		uint32_t* d_helper_pointer, *d_helper_position;
		cudaMalloc(&d_helper_pointer, sizeof(uint32_t) * (matTransposeOut.rows + 1));
		cudaMalloc(&d_helper_position, sizeof(uint32_t) * (matTransposeOut.nnz));
		cudaMemset(d_helper_pointer, 0, sizeof(uint32_t) * (matTransposeOut.rows + 1));
		cudaMemset(matTransposeOut.row_offsets, 0, (matTransposeOut.rows + 1) * sizeof(uint32_t));

		// Calculate entry distribution
		d_calulateTransposeDistribution<<<gridSize , blockSize >>>(matIn.rows, matIn.cols, matIn.row_offsets, matIn.col_ids, matTransposeOut.row_offsets);

		// Prefix sum for new offset vector
		thrust::device_ptr<uint32_t> th_offset_vector(matTransposeOut.row_offsets);
		thrust::exclusive_scan(th_offset_vector, th_offset_vector + matTransposeOut.rows + 1, th_offset_vector);

		// Find position for insertion (keeping sort order)
		d_findPosition<DataType> << <gridSize, blockSize >> > (matIn.rows, matIn.cols, matIn.row_offsets, matIn.col_ids, matIn.data, matTransposeOut.row_offsets, matTransposeOut.col_ids, matTransposeOut.data, d_helper_pointer, d_helper_position);

		// Write Transpose
		d_writeTranspose<DataType> << <gridSize, blockSize >> > (matIn.rows, matIn.cols, matIn.row_offsets, matIn.col_ids, matIn.data, matTransposeOut.row_offsets, matTransposeOut.col_ids, matTransposeOut.data, d_helper_pointer, d_helper_position);

		// Free helper resources
		cudaFree(d_helper_pointer);
		cudaFree(d_helper_position);

		return;
	}

	template void Transpose<float>(const dCSR<float>& matIn, dCSR<float>& matTransposeOut);
	template void Transpose<double>(const dCSR<double>& matIn, dCSR<double>& matTransposeOut);
}