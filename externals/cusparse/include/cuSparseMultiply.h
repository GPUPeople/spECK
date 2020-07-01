#pragma once

#include "dCSR.h"
#include <cusparse.h>
#include <iostream>
#include <string>

namespace cuSPARSE {

	template <typename DataType>
	class CuSparseTest
	{
		cusparseHandle_t handle;
		cusparseStatus_t status;
		cusparseMatDescr_t descr;
		cusparseMatDescr_t descrB;
		cusparseMatDescr_t descrC;

	public:
		CuSparseTest(): handle(0)
		{
			checkCuSparseError(cusparseCreate(&handle), "init failed");
			checkCuSparseError(cusparseCreateMatDescr(&descr), "Matrix descriptor init failed");
			checkCuSparseError(cusparseCreateMatDescr(&descrB), "Matrix descriptor init failed");
			checkCuSparseError(cusparseCreateMatDescr(&descrC), "Matrix descriptor init failed");
			cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
			cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
			cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL);
			cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO);
			cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
			cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
		}

		~CuSparseTest()
		{
			checkCuSparseError(cusparseDestroyMatDescr(descr), "Matrix descriptor destruction failed");
			checkCuSparseError(cusparseDestroyMatDescr(descrB), "Matrix descriptor destruction failed");
			checkCuSparseError(cusparseDestroyMatDescr(descrC), "Matrix descriptor destruction failed");
			cusparseDestroy(handle);
		}

		// Multiply two CSR matrices
		float Multiply(const dCSR<DataType>& A, const dCSR<DataType>& B, dCSR<DataType>& matOut, uint32_t& cusparse_nnz);

		void Transpose(const dCSR<DataType>& A, dCSR<DataType>& AT);

		cusparseStatus_t checkCuSparseError(cusparseStatus_t status, std::string errorMsg)
		{
			if (status != CUSPARSE_STATUS_SUCCESS) {
				std::cout << "CuSparse error: " << errorMsg << std::endl;
				throw std::exception();
			}
			return status;
		}

		cusparseStatus_t CUSPARSEAPI cusparseMultiply(cusparseHandle_t handle,
			cusparseOperation_t transA,
			cusparseOperation_t transB,
			int m,
			int n,
			int k,
			const cusparseMatDescr_t descrA,
			int nnzA,
			const DataType *csrSortedValA,
			const int *csrSortedRowPtrA,
			const int *csrSortedColIndA,
			const cusparseMatDescr_t descrB,
			int nnzB,
			const DataType *csrSortedValB,
			const int *csrSortedRowPtrB,
			const int *csrSortedColIndB,
			const cusparseMatDescr_t descrC,
			DataType *csrSortedValC,
			const int *csrSortedRowPtrC,
			int *csrSortedColIndC);

		cusparseStatus_t CUSPARSEAPI cusparseTranspose(cusparseHandle_t handle,
			int m,
			int n,
			int nnz,
			const DataType  *csrSortedVal,
			const int *csrSortedRowPtr,
			const int *csrSortedColInd,
			DataType *cscSortedVal,
			int *cscSortedRowInd,
			int *cscSortedColPtr,
			cusparseAction_t copyValues,
			cusparseIndexBase_t idxBase);		
	};	
}