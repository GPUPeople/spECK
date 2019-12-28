#include "cusparse/include/cuSparseMultiply.h"
#include <cuda_runtime.h>


namespace cuSPARSE {
		template<>
		cusparseStatus_t CUSPARSEAPI CuSparseTest<double>::cusparseMultiply(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB,
			int m, int n, int k, const cusparseMatDescr_t descrA, int nnzA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
			const cusparseMatDescr_t descrB, int nnzB, const double *csrSortedValB, const int *csrSortedRowPtrB, const int *csrSortedColIndB,
			const cusparseMatDescr_t descrC, double *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC){
			return cusparseDcsrgemm(handle, transA, transB, m, n, k,
				descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
				descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB,
				descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
		}

		template<>
		cusparseStatus_t CUSPARSEAPI CuSparseTest<float>::cusparseMultiply(cusparseHandle_t handle, cusparseOperation_t transA,	cusparseOperation_t transB,	
			int m, int n, int k, const cusparseMatDescr_t descrA, int nnzA, const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,
			const cusparseMatDescr_t descrB, int nnzB, const float *csrSortedValB, const int *csrSortedRowPtrB,	const int *csrSortedColIndB,
			const cusparseMatDescr_t descrC, float *csrSortedValC, const int *csrSortedRowPtrC,	int *csrSortedColIndC){
			return cusparseScsrgemm(handle, transA, transB, m, n, k,
				descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
				descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB,
				descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
		}

		template<>
		cusparseStatus_t CUSPARSEAPI CuSparseTest<float>::cusparseTranspose(cusparseHandle_t handle, int m, int n, int nnz,
			const float  *csrSortedVal,	const int *csrSortedRowPtr, const int *csrSortedColInd,
			float *cscSortedVal, int *cscSortedRowInd, int *cscSortedColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase)
		{
			return cusparseScsr2csc(handle, m, n, nnz, csrSortedVal, csrSortedRowPtr, csrSortedColInd, cscSortedVal,
				cscSortedRowInd, cscSortedColPtr, copyValues, idxBase);
		}

		template<>
		cusparseStatus_t CUSPARSEAPI CuSparseTest<double>::cusparseTranspose(cusparseHandle_t handle,  int m, int n, int nnz,
			const double  *csrSortedVal, const int *csrSortedRowPtr, const int *csrSortedColInd,
			double *cscSortedVal, int *cscSortedRowInd, int *cscSortedColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase)
		{
			return cusparseDcsr2csc(handle, m, n, nnz, csrSortedVal, csrSortedRowPtr, csrSortedColInd, cscSortedVal,
				cscSortedRowInd, cscSortedColPtr, copyValues, idxBase);
		}

	template <typename DataType>
	float CuSparseTest<DataType>::Multiply(const dCSR<DataType>& A, const dCSR<DataType>& B, dCSR<DataType>& matOut, uint32_t& cusparse_nnz)
	{
		int nnzC;
		int *nnzTotalDevHostPtr = &nnzC;
		float duration;
		int m, n, k;
		m = A.rows;
		n = B.cols;
		k = A.cols;
		// matOut.reset();

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		// ############################
		cudaEventRecord(start);
		// ############################

		// Allocate memory for row indices
		if(matOut.rows != A.rows)
		{
			if (matOut.row_offsets != nullptr)
				cudaFree(matOut.row_offsets);

			cudaMalloc(&(matOut.row_offsets), sizeof(uint32_t) * (A.rows + 1));
		}

		// Precompute number of nnz in C
		checkCuSparseError(cusparseXcsrgemmNnz(
			handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			m, n, k,
			descr, A.nnz, reinterpret_cast<const int*>(A.row_offsets), reinterpret_cast<const int*>(A.col_ids),
			descrB, B.nnz, reinterpret_cast<const int*>(B.row_offsets), reinterpret_cast<const int*>(B.col_ids),
			descrC, reinterpret_cast<int*>(matOut.row_offsets), nnzTotalDevHostPtr), "cuSparse: Precompute failed"
		);

		cusparse_nnz = nnzC;

		// Allocate rest of memory
		if(nnzC != matOut.nnz)
		{
			if (matOut.col_ids != nullptr)
				cudaFree(matOut.col_ids);
			if (matOut.data != nullptr)
				cudaFree(matOut.data);

			cudaMalloc(&(matOut.col_ids), sizeof(uint32_t) * nnzC);
			cudaMalloc(&(matOut.data), sizeof(DataType) * nnzC);
		}
		
		// Compute SpGEMM
		checkCuSparseError(cusparseMultiply(
			handle,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			m, n, k,
			descr, A.nnz, reinterpret_cast<const DataType*>(A.data), reinterpret_cast<const int*>(A.row_offsets), reinterpret_cast<const int*>(A.col_ids),
			descrB, B.nnz, reinterpret_cast<const DataType*>(B.data), reinterpret_cast<const int*>(B.row_offsets), reinterpret_cast<const int*>(B.col_ids),
			descrC, reinterpret_cast<DataType*>(matOut.data), reinterpret_cast<int*>(matOut.row_offsets), reinterpret_cast<int*>(matOut.col_ids)),
			"cuSparse: SpGEMM failed");

		matOut.nnz = nnzC;
		matOut.rows = m;
		matOut.cols = n;

		// ############################
		cudaDeviceSynchronize();
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		// ############################

		cudaEventElapsedTime(&duration, start, stop);

		return duration;
	}

	template float CuSparseTest<float>::Multiply(const dCSR<float>& A, const dCSR<float>& B, dCSR<float>& matOut, uint32_t& cusparse_nnz);
	template float CuSparseTest<double>::Multiply(const dCSR<double>& A, const dCSR<double>& B, dCSR<double>& matOut, uint32_t& cusparse_nnz);

	template <typename DataType>
	void CuSparseTest<DataType>::Transpose(const dCSR<DataType>& A, dCSR<DataType>& AT)
	{
		AT.alloc(A.cols, A.rows, A.nnz);

		checkCuSparseError(cusparseTranspose(handle, A.rows, A.cols, A.nnz,
			reinterpret_cast<const DataType*>(A.data), reinterpret_cast<const int*>(A.row_offsets), reinterpret_cast<const int*>(A.col_ids),
			reinterpret_cast<DataType*>(AT.data), reinterpret_cast<int*>(AT.col_ids), reinterpret_cast<int*>(AT.row_offsets),
			CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO), "transpose failed");
	}

	template	void CuSparseTest<float>::Transpose(const dCSR<float>& A, dCSR<float>& AT);
	template	void CuSparseTest<double>::Transpose(const dCSR<double>& A, dCSR<double>& AT);
}