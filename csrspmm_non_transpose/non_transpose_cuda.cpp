#include "cuda_util.cuh"
#include "gespmm.h"
#include <torch/extension.h>

//CUDA declarations

void csrspmm_non_transpose_parreduce_rowbalance_cuda(const SpMatCsrDescr_t spmatA,
                                                const float *B, const int N,
                                                float *C);


//C++ interface
void csrspmm_non_transpose_parreduce_rowbalance(const SpMatCsrDescr_t spmatA,
                                                const float *B, const int N,
                                                float *C){
    return csrspmm_non_transpose_parreduce_rowbalance_cuda(spmatA, B, N,C);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("csrspmm_non_transpose_parreduce_rowbalance", &csrspmm_non_transpose_parreduce_rowbalance, "csrspmm_non_transpose_parreduce_rowbalance(CUDA)");
}

