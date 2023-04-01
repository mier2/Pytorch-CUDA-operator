#include "include/cuda_util.cuh"
#include "include/gespmm.h"
#include <torch/extension.h>
#include <iostream>

void csrspmm_non_transpose_parreduce_nnzbalance_cuda(const SpMatCsrDescr_t spmatA,
                                                const float *B, const int N,
                                                float *C);

void csrspmm_non_transpose_parreduce_rowbalance_cuda(const SpMatCsrDescr_t spmatA,
                                                const float *B, const int N,
                                                float *C);

void csrspmm_non_transpose_seqreduce_nnzbalance_cuda(const SpMatCsrDescr_t spmatA,
                                                const float *B, const int N,
                                                float *C);


void csrspmm_non_transpose_seqreduce_rowbalance_cuda(const SpMatCsrDescr_t spmatA,
                                                const float *B, const int N,
                                                float *C);



void csrspmm_parreduce_nnzbalance_cuda(const SpMatCsrDescr_t spmatA,
                                                const float *B, const int N,
                                                float *C);

void csrspmm_parreduce_rowbalance_cuda(const SpMatCsrDescr_t spmatA,
                                                const float *B, const int N,
                                                float *C);

void csrspmm_rowcaching_nnzbalance_cuda(const SpMatCsrDescr_t spmatA,
                                                const float *B, const int N,
                                                float *C);

void csrspmm_rowcaching_rowbalance_cuda(const SpMatCsrDescr_t spmatA,
                                                const float *B, const int N,
                                                float *C);

void csrspmm_seqreduce_nnzbalance_cuda(const SpMatCsrDescr_t spmatA,
                                                const float *B, const int N,
                                                float *C);

void csrspmm_seqreduce_rowbalance_cuda(const SpMatCsrDescr_t spmatA,
                                                const float *B, const int N,
                                                float *C);

//C++ interface
void csrspmm_seqreduce_rowbalance(int nrow, int ncol, int nnz,
                                                torch::Tensor indptr, torch::Tensor indices, torch::Tensor data,
                                                const torch::Tensor B, const int N, torch::Tensor C){
    const SpMatCsrDescr_t spmatA = {nrow, ncol, nnz, indptr.data_ptr<int>(), indices.data_ptr<int>(), data.data_ptr<float>()};

    return csrspmm_seqreduce_rowbalance_cuda(spmatA, B.data_ptr<float>(), N, C.data_ptr<float>());
}

void csrspmm_seqreduce_nnzbalance(int nrow, int ncol, int nnz,
                                                torch::Tensor indptr, torch::Tensor indices, torch::Tensor data,
                                                const torch::Tensor B, const int N, torch::Tensor C){
    const SpMatCsrDescr_t spmatA = {nrow, ncol, nnz, indptr.data_ptr<int>(), indices.data_ptr<int>(), data.data_ptr<float>()};

    return csrspmm_seqreduce_nnzbalance_cuda(spmatA, B.data_ptr<float>(), N, C.data_ptr<float>());
}

void csrspmm_rowcaching_rowbalance(int nrow, int ncol, int nnz,
                                                torch::Tensor indptr, torch::Tensor indices, torch::Tensor data,
                                                const torch::Tensor B, const int N, torch::Tensor C){
    const SpMatCsrDescr_t spmatA = {nrow, ncol, nnz, indptr.data_ptr<int>(), indices.data_ptr<int>(), data.data_ptr<float>()};

    return csrspmm_rowcaching_rowbalance_cuda(spmatA, B.data_ptr<float>(), N, C.data_ptr<float>());
}

void csrspmm_rowcaching_nnzbalance(int nrow, int ncol, int nnz,
                                                torch::Tensor indptr, torch::Tensor indices, torch::Tensor data,
                                                const torch::Tensor B, const int N, torch::Tensor C){
    const SpMatCsrDescr_t spmatA = {nrow, ncol, nnz, indptr.data_ptr<int>(), indices.data_ptr<int>(), data.data_ptr<float>()};

    return csrspmm_rowcaching_nnzbalance_cuda(spmatA, B.data_ptr<float>(), N, C.data_ptr<float>());
}

void csrspmm_parreduce_rowbalance(int nrow, int ncol, int nnz,
                                                torch::Tensor indptr, torch::Tensor indices, torch::Tensor data,
                                                const torch::Tensor B, const int N, torch::Tensor C){
    const SpMatCsrDescr_t spmatA = {nrow, ncol, nnz, indptr.data_ptr<int>(), indices.data_ptr<int>(), data.data_ptr<float>()};

    return csrspmm_parreduce_rowbalance_cuda(spmatA, B.data_ptr<float>(), N, C.data_ptr<float>());
}

void csrspmm_parreduce_nnzbalance(int nrow, int ncol, int nnz,
                                                torch::Tensor indptr, torch::Tensor indices, torch::Tensor data,
                                                const torch::Tensor B, const int N, torch::Tensor C){
    const SpMatCsrDescr_t spmatA = {nrow, ncol, nnz, indptr.data_ptr<int>(), indices.data_ptr<int>(), data.data_ptr<float>()};

    return csrspmm_parreduce_nnzbalance_cuda(spmatA, B.data_ptr<float>(), N, C.data_ptr<float>());
}


void csrspmm_non_transpose_seqreduce_rowbalance(int nrow, int ncol, int nnz,
                                                torch::Tensor indptr, torch::Tensor indices, torch::Tensor data,
                                                const torch::Tensor B, const int N, torch::Tensor C){
    const SpMatCsrDescr_t spmatA = {nrow, ncol, nnz, indptr.data_ptr<int>(), indices.data_ptr<int>(), data.data_ptr<float>()};

    return csrspmm_non_transpose_seqreduce_nnzbalance_cuda(spmatA, B.data_ptr<float>(), N, C.data_ptr<float>());
}

void csrspmm_non_transpose_seqreduce_nnzbalance(int nrow, int ncol, int nnz,
                                                torch::Tensor indptr, torch::Tensor indices, torch::Tensor data,
                                                const torch::Tensor B, const int N, torch::Tensor C){
    const SpMatCsrDescr_t spmatA = {nrow, ncol, nnz, indptr.data_ptr<int>(), indices.data_ptr<int>(), data.data_ptr<float>()};

    return csrspmm_non_transpose_seqreduce_nnzbalance_cuda(spmatA, B.data_ptr<float>(), N, C.data_ptr<float>());
}


void csrspmm_non_transpose_parreduce_rowbalance(int nrow, int ncol, int nnz,
                                                torch::Tensor indptr, torch::Tensor indices, torch::Tensor data,
                                                const torch::Tensor B, const int N, torch::Tensor C){
    const SpMatCsrDescr_t spmatA = {nrow, ncol, nnz, indptr.data_ptr<int>(), indices.data_ptr<int>(), data.data_ptr<float>()};

    return csrspmm_non_transpose_parreduce_rowbalance_cuda(spmatA, B.data_ptr<float>(), N, C.data_ptr<float>());
}

void csrspmm_non_transpose_parreduce_nnzbalance(int nrow, int ncol, int nnz,
                                                torch::Tensor indptr, torch::Tensor indices, torch::Tensor data,
                                                const torch::Tensor B, const int N, torch::Tensor C){
    const SpMatCsrDescr_t spmatA = {nrow, ncol, nnz, indptr.data_ptr<int>(), indices.data_ptr<int>(), data.data_ptr<float>()};

    return csrspmm_non_transpose_parreduce_nnzbalance_cuda(spmatA, B.data_ptr<float>(), N, C.data_ptr<float>());
}






PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("csrspmm_non_transpose_parreduce_nnzbalance", &csrspmm_non_transpose_parreduce_nnzbalance, "csrspmm_non_transpose_parreduce_nnzbalance wrapper");
  m.def("csrspmm_non_transpose_parreduce_rowbalance", &csrspmm_non_transpose_parreduce_rowbalance, "csrspmm_non_transpose_parreduce_rowbalance wrapper");
  m.def("csrspmm_non_transpose_seqreduce_nnzbalance", &csrspmm_non_transpose_seqreduce_nnzbalance, "csrspmm_non_transpose_seqreduce_nnzbalance wrapper");
  m.def("csrspmm_non_transpose_seqreduce_rowbalance", &csrspmm_non_transpose_seqreduce_nnzbalance, "csrspmm_non_transpose_seqreduce_nnzbalance wrapper");
  m.def("csrspmm_parreduce_nnzbalance", &csrspmm_parreduce_nnzbalance, "csrspmm_parreduce_nnzbalance wrapper");
  m.def("csrspmm_parreduce_rowbalance", &csrspmm_parreduce_rowbalance, "csrspmm_parreduce_rowbalance wrapper");
  m.def("csrspmm_rowcaching_nnzbalance", &csrspmm_rowcaching_nnzbalance, "csrspmm_rowcaching_nnzbalance wrapper");
  m.def("csrspmm_rowcaching_rowbalance", &csrspmm_rowcaching_rowbalance, "csrspmm_rowcaching_rowbalance wrapper");
  m.def("csrspmm_seqreduce_nnzbalance", &csrspmm_seqreduce_nnzbalance, "csrspmm_seqreduce_nnzbalance wrapper");
  m.def("csrspmm_seqreduce_rowbalance", &csrspmm_seqreduce_rowbalance, "csrspmm_seqreduce_rowbalance wrapper");
}