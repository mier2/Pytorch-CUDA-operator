#include "include/cuda_util.cuh"
#include "include/gespmm.h"
#include <torch/extension.h>


void csrspmm_non_transpose_parreduce_rowbalance_cuda(const SpMatCsrDescr_t spmatA,
                                                const float *B, const int N,
                                                float *C);


//C++ interface
void csrspmm_non_transpose_parreduce_rowbalance(int nrow, int ncol, int nnz,
                                                torch::Tensor &indptr, torch::Tensor &indices, torch::Tensor &data,
                                                const torch::Tensor &B, const int N, torch::Tensor &C){
    const SpMatCsrDescr_t spmatA = {nrow, ncol, nnz, (int *)indptr.data_ptr(), (int *)indices.data_ptr(), (float *)data.data_ptr()};


    return csrspmm_non_transpose_parreduce_rowbalance_cuda(spmatA, (const float *)B.data_ptr(), N,(float *)C.data_ptr());
}





PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("csrspmm_non_transpose_parreduce_rowbalance", &csrspmm_non_transpose_parreduce_rowbalance, "csrspmm_non_transpose_parreduce_rowbalance wrapper");
}

