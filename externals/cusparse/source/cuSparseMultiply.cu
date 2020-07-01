#include "cusparse/include/cuSparseMultiply.h"
#include <cuda_runtime.h>
#include "common.h"


namespace cuSPARSE {
		template<>
		cusparseStatus_t CUSPARSEAPI CuSparseTest<float>::cusparseTranspose(cusparseHandle_t handle, int m, int n, int nnz,
			const float  *csrSortedVal,	const int *csrSortedRowPtr, const int *csrSortedColInd,
			float *cscSortedVal, int *cscSortedRowInd, int *cscSortedColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase)
		{
            void *buffer = nullptr;
            size_t buffer_size = 0;
            checkCuSparseError(cusparseCsr2cscEx2_bufferSize(handle, m, n, nnz, csrSortedVal, csrSortedRowPtr, csrSortedColInd, cscSortedVal,
				cscSortedColPtr, cscSortedRowInd, CUDA_R_32F, copyValues, idxBase, CUSPARSE_CSR2CSC_ALG1, &buffer_size), "buffer size failed");
            HANDLE_ERROR(cudaMalloc(&buffer, buffer_size));

            auto retVal = checkCuSparseError(cusparseCsr2cscEx2(handle, m, n, nnz, csrSortedVal, csrSortedRowPtr, csrSortedColInd, cscSortedVal,
				cscSortedColPtr, cscSortedRowInd, CUDA_R_32F, copyValues, idxBase, CUSPARSE_CSR2CSC_ALG1, buffer), "transpose failed");
            HANDLE_ERROR(cudaFree(buffer));
            return retVal;
		}

		template<>
		cusparseStatus_t CUSPARSEAPI CuSparseTest<double>::cusparseTranspose(cusparseHandle_t handle,  int m, int n, int nnz,
			const double  *csrSortedVal, const int *csrSortedRowPtr, const int *csrSortedColInd,
			double *cscSortedVal, int *cscSortedRowInd, int *cscSortedColPtr, cusparseAction_t copyValues, cusparseIndexBase_t idxBase)
		{
            void *buffer = nullptr;
            size_t buffer_size = 0;
            checkCuSparseError(cusparseCsr2cscEx2_bufferSize(handle, m, n, nnz, csrSortedVal, csrSortedRowPtr, csrSortedColInd, cscSortedVal,
				cscSortedColPtr, cscSortedRowInd, CUDA_R_64F, copyValues, idxBase, CUSPARSE_CSR2CSC_ALG1, &buffer_size), "buffer size failed");
            HANDLE_ERROR(cudaDeviceSynchronize());
            HANDLE_ERROR(cudaMalloc(&buffer, buffer_size));

            auto retVal = checkCuSparseError(cusparseCsr2cscEx2(handle, m, n, nnz, csrSortedVal, csrSortedRowPtr, csrSortedColInd, cscSortedVal,
				cscSortedColPtr, cscSortedRowInd, CUDA_R_64F, copyValues, idxBase, CUSPARSE_CSR2CSC_ALG1, buffer), "transpose failed");
            HANDLE_ERROR(cudaFree(buffer));
            return retVal;
		}

	template <typename DataType>
	float CuSparseTest<DataType>::Multiply(const dCSR<DataType>& A, const dCSR<DataType>& B, dCSR<DataType>& matOut, uint32_t& cusparse_nnz)
	{
		int nnzC;
		int *nnzTotalDevHostPtr = &nnzC;
		float duration;
        DataType alpha = (DataType) 1.0f;
        DataType beta = (DataType) 0.0f;

		cudaEvent_t start, stop;
		HANDLE_ERROR(cudaEventCreate(&start));
		HANDLE_ERROR(cudaEventCreate(&stop));

		// ############################
		HANDLE_ERROR(cudaEventRecord(start));
		// ############################

        auto computeType = sizeof(DataType) == 4 ? CUDA_R_32F : CUDA_R_64F;
        cusparseSpMatDescr_t matA, matB, matC;
        checkCuSparseError( cusparseCreateCsr(&matA, A.rows, A.cols, A.nnz,
                                        A.row_offsets, A.col_ids, A.data,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, computeType), "A failed");
        checkCuSparseError( cusparseCreateCsr(&matB, B.rows, B.cols, B.nnz,
                                        B.row_offsets, B.col_ids, B.data,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, computeType), "B failed");
        checkCuSparseError( cusparseCreateCsr(&matC, A.rows, B.cols, 0,
                                        NULL, NULL, NULL,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, computeType), "C failed");

        void*  dBuffer1    = NULL, *dBuffer2   = NULL;
        size_t bufferSize1 = 0,    bufferSize2 = 0;
        cusparseSpGEMMDescr_t spgemmDesc;
        checkCuSparseError( cusparseSpGEMM_createDescr(&spgemmDesc), "create description failed");
        auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
        auto opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
        // Device memory management: Allocate and copy A, B
        int   *dC_csrOffsets = nullptr, *dC_columns = nullptr;
        DataType *dC_values;

        // ask bufferSize1 bytes for external memory
        checkCuSparseError(cusparseSpGEMM_workEstimation(handle, opA, opB,
                                    &alpha, matA, matB, &beta, matC,
                                    computeType, CUSPARSE_SPGEMM_DEFAULT,
                                    spgemmDesc, &bufferSize1, 0), "workestimation0 failed");
        HANDLE_ERROR(cudaMalloc((void**) &dBuffer1, bufferSize1));
        // inspect the matrices A and B to understand the memory requirement for
        // the next step
        checkCuSparseError(cusparseSpGEMM_workEstimation(handle, opA, opB,
                                    &alpha, matA, matB, &beta, matC,
                                    computeType, CUSPARSE_SPGEMM_DEFAULT,
                                    spgemmDesc, &bufferSize1, dBuffer1), "workestimation1 failed");

        // ask bufferSize2 bytes for external memory
        checkCuSparseError(cusparseSpGEMM_compute(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT,
                            spgemmDesc, &bufferSize2, NULL), "compute0 failed");
        HANDLE_ERROR(cudaMalloc((void**) &dBuffer2, bufferSize2));

        // compute the intermediate product of A * B
        checkCuSparseError(cusparseSpGEMM_compute(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT,
                            spgemmDesc, &bufferSize2, dBuffer2), "compute1 failed");
        // get matrix C non-zero entries C_num_nnz1
        int64_t C_num_rows1, C_num_cols1, C_num_nnz1;
         checkCuSparseError(cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_num_nnz1), "get size failed");
        // allocate matrix C
        HANDLE_ERROR(cudaMalloc((void**) &dC_csrOffsets, (C_num_rows1 + 1) * sizeof(int)));
        HANDLE_ERROR(cudaMalloc((void**) &dC_columns, C_num_nnz1 * sizeof(int)));
        HANDLE_ERROR(cudaMalloc((void**) &dC_values,  C_num_nnz1 * sizeof(DataType)));
        // update matC with the new pointers
        checkCuSparseError(cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values), "get pointers failed");

        // copy the final products to the matrix C
        checkCuSparseError(cusparseSpGEMM_copy(
            handle, 
            opA, 
            opB,
            &alpha, 
            matA, 
            matB, 
            &beta, 
            matC,
            computeType, 
            CUSPARSE_SPGEMM_DEFAULT, 
            spgemmDesc), 
            "copy failed");

        cusparseIndexType_t _rowType, _columnType;
        cusparseIndexBase_t _indexBase;
        cudaDataType _baseOff;
        checkCuSparseError(cusparseCsrGet(matC,
            (int64_t*) &matOut.rows,
            (int64_t*) &matOut.cols,
            (int64_t*) &matOut.nnz,
            (void**) &matOut.row_offsets,
            (void**) &matOut.col_ids,
            (void**) &matOut.data,
            &_rowType,
            &_columnType,
            &_indexBase,
            &_baseOff
        ), "get failed");
        // destroy matrix/vector descriptors
        checkCuSparseError( cusparseSpGEMM_destroyDescr(spgemmDesc), "destroy failed" );
        HANDLE_ERROR(cudaFree(dBuffer1));
        HANDLE_ERROR(cudaFree(dBuffer2));

		// ############################
		HANDLE_ERROR(cudaDeviceSynchronize());
		HANDLE_ERROR(cudaEventRecord(stop));
		HANDLE_ERROR(cudaEventSynchronize(stop));
		// ############################

		HANDLE_ERROR(cudaEventElapsedTime(&duration, start, stop));
        cusparse_nnz = matOut.nnz;

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