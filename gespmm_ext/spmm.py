import sys
import numpy as np
import torch
import scipy
import gespmm_ext as csrspmm
import argparse
from data_loader import DataLoader


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

        csrspmm.csrspmm_seqreduce_rowbalance(M, K, A_csr.nnz, indptr_tensor, indices_tensor, data_tensor, B, N, C)

        if torch.all(torch.abs(C-C_ref) > 1e-2* torch.abs(C_ref)):
                print("i am watching you mike wazowski")
        else:
                print("Fuck yeah")
        

        print("C")
        print(C)

        print("C-ref")
        print(C_ref)




if __name__ == "__main__":
        main(sys.argv)