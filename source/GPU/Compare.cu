// Global includes
#include <stdio.h>
#include <stdint.h>

// Local includes
#include "Compare.h"
#include "common.h"

#define VERIFICATION_TEXT

template <typename DataType>
__global__ void d_compare(int in_rows, int in_cols, const uint32_t* __restrict reference_offset, const uint32_t* __restrict reference_indices, const DataType* __restrict reference_values,
	const uint32_t* __restrict compare_offset, const uint32_t* __restrict compare_indices, const DataType* __restrict compare_values, bool compare_data, double epsilon, uint32_t* verification)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= in_rows)
		return;

	// if (tid > 10000)
	// 	return;

	uint32_t ref_offset = reference_offset[tid];
	uint32_t comp_offset = compare_offset[tid];
	uint32_t ref_number_entries = reference_offset[tid + 1] - ref_offset;
	uint32_t comp_number_entries = compare_offset[tid + 1] - comp_offset;

	if (ref_number_entries != comp_number_entries)
	{
#ifdef VERIFICATION_TEXT
		printf("---------- Row: %u | Row length not identical: (Ref|Comp) : (%u|%u)\n",tid, ref_number_entries, comp_number_entries);
#endif
		*verification = 1;
		return;
	}

	uint32_t num_entries = min(ref_number_entries, comp_number_entries);

	for (uint32_t i = 0; i < num_entries; ++i)
	{
		if (reference_indices[ref_offset + i] != compare_indices[comp_offset + i])
		{
#ifdef VERIFICATION_TEXT
			printf("Row: %u | Row indices do NOT match: (Ref|Comp) : (%u|%u) - pos: %u/%u\n", tid, reference_indices[ref_offset + i], compare_indices[comp_offset + i], i, num_entries);
#endif
			*verification = 1;
			return;
		}
		if (compare_data)
		{
			if (compare_values[comp_offset + i] != 0 && std::abs(reference_values[ref_offset + i] / compare_values[comp_offset + i] - 1) > 0.01)
			{
#ifdef VERIFICATION_TEXT
				printf("Row: %u | Values do NOT match: (Ref|Comp) : (%f|%f) - pos: %u/%u - col %u\n", tid, reference_values[ref_offset + i], compare_values[comp_offset + i], i, num_entries, reference_indices[ref_offset + i]);
#endif
				*verification = 1;
				// return;
			}
		}
	}

	return;
}

namespace spECK {
	template <typename DataType>
	bool Compare(const dCSR<DataType>& reference_mat, const dCSR<DataType>& compare_mat, bool compare_data)
	{
		int blockSize(256);
		int gridSize(divup<int>(reference_mat.rows + 1, blockSize));
		double epsilon = 0.1;
		uint32_t* verification, h_verification;
		cudaMalloc(&verification, sizeof(uint32_t));
		cudaMemset(verification, 0, sizeof(uint32_t));

		d_compare<DataType> << <gridSize, blockSize >> > (reference_mat.rows, reference_mat.cols,
			reference_mat.row_offsets, reference_mat.col_ids, reference_mat.data,
			compare_mat.row_offsets, compare_mat.col_ids, compare_mat.data,
			compare_data, epsilon, verification);
		 
		cudaMemcpy(&h_verification, verification, sizeof(uint32_t), cudaMemcpyDeviceToHost);
		return (h_verification == 0);
	}

	template bool Compare<float>(const dCSR<float>& reference_mat, const dCSR<float>& compare_mat, bool compare_data);
	template bool Compare<double>(const dCSR<double>& reference_mat, const dCSR<double>& compare_mat, bool compare_data);
}