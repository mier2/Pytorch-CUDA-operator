#include "include/cuda_util.cuh"
#include "include/gespmm.h"
#include <torch/extension.h>
#include <iostream>


void csrspmm_non_transpose_parreduce_rowbalance_cuda(const SpMatCsrDescr_t spmatA,
                                                const float *B, const int N,
                                                float *C);


//C++ interface
void csrspmm_non_transpose_parreduce_rowbalance(int nrow, int ncol, int nnz,
                                                torch::Tensor indptr, torch::Tensor indices, torch::Tensor data,
                                                const torch::Tensor B, const int N, torch::Tensor C){
    const SpMatCsrDescr_t spmatA = {nrow, ncol, nnz, indptr.data_ptr<int>(), indices.data_ptr<int>(), data.data_ptr<float>()};

    return csrspmm_non_transpose_parreduce_rowbalance_cuda(spmatA, B.data_ptr<float>(), N, C.data_ptr<float>());
}






PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("csrspmm_non_transpose_parreduce_rowbalance", &csrspmm_non_transpose_parreduce_rowbalance, "csrspmm_non_transpose_parreduce_rowbalance wrapper");
}

