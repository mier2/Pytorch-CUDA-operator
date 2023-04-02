import sys
import numpy as np
import torch
import scipy
import time
from spmm_util import SpmmUtil, DataLoader, GpuTimer


def main(argv):
        if len(argv) < 2:
                print("Require command-line argument: name of the sparse matrix file in .mtx format.")
                return
        #load sparse matrix A in csr format
        filepath = "../matrices/"+argv[1]
        data_loader = DataLoader(filepath)
        A_sparse = data_loader.read_mtx()
        A_csr = scipy.sparse.csr_matrix(A_sparse)
        M,K = A_csr.shape
        print(f"Finish reading matrix {M} rows, {K} columns, {A_csr.nnz} nnzs")

        #Genrate arrays
        N = 128 # number of B colums
        if len(argv) >2:
                N = int(argv[2])
                print("second command-line argument is number of B columns, should be >0.")
        
        B = torch.zeros((K, N), dtype=torch.float32).cuda()
        C = torch.zeros((M, N), dtype=torch.float32).cuda()
        B.uniform_(0,0.2)

        A_sparse_tensor = torch.sparse_coo_tensor(np.vstack((A_sparse.row, A_sparse.col)), A_sparse.data, A_sparse.shape).cuda()
        A_sparse_tensor = A_sparse_tensor.to(torch.float32).cuda()

        C_ref = torch.sparse.mm(A_sparse_tensor, B).cuda()
        indptr_tensor = torch.tensor(A_csr.indptr, dtype=torch.int32).cuda()
        indices_tensor = torch.tensor(A_csr.indices, dtype=torch.int32).cuda()
        data_tensor = torch.tensor(A_csr.data, dtype=torch.float32).cuda()
        
        #select the algorithm and check the result
        alg = "GESPMM_ALG_DEFAULT"
        spmm_util = SpmmUtil()
        spmm_util.gespmmCsrSpMM(M, K, A_csr.nnz, indptr_tensor, indices_tensor, data_tensor, B, N,C, True, alg)
        correct = spmm_util.check_result(C_ref, C)

        if correct:
                gpu_timer = GpuTimer()
                warmup_iter = 10
                repeat_iter = 100
                for iter in range(warmup_iter+repeat_iter):
                        if iter == warmup_iter:
                                gpu_timer.start()
                        spmm_util.gespmmCsrSpMM(M, K, A_csr.nnz, indptr_tensor, indices_tensor, data_tensor, B, N,C, True, alg)
                gpu_timer.stop()
                
                kernel_dur_msecs = gpu_timer.elapsed_time()
                MFlop_count = A_csr.nnz/1e6 * N * 2
                gflops = MFlop_count/kernel_dur_msecs
                sparsity = A_csr.nnz/M/K
                
                print(f"[GE-SpMM][Alg: {alg}] Report: spmm A({M} x {K}) * B({K} x {N}) sparsity {sparsity} (nnz={A_csr.nnz}) \n Time {kernel_dur_msecs} (ms), Throughput {gflops} (gflops).")
                
                        

        




if __name__ == "__main__":
        main(sys.argv)