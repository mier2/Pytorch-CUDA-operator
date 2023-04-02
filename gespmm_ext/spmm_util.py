import sys
import numpy as np
import torch
import scipy
import gespmm_ext as csrspmm


class SpmmUtil:
    def __inti__(self):
           pass
    #check result
    def check_result(self, C_ref, C):
        if torch.all(torch.abs(C-C_ref) > 1e-2* torch.abs(C_ref)):
                print("Wrong answer")
        else:
                print("Passed")

    #algo selector
    def gespmmAlgSel(self,dense_ncol, transpose_BC):
        if transpose_BC:
                if dense_ncol >= 32:
                        return "GESPMM_ALG_ROWCACHING_ROWBALANCE"
                elif dense_ncol >4:
                        return "GESPMM_ALG_SEQREDUCE_ROWBALANCE"
                else:
                        return "GESPMM_ALG_PARREDUCE_ROWBALANCE_NON_TRANSPOSE"
    
    #algo dispatcher
    def gespmmCsrSpMM(self, M, K, nnz, indptr_tensor, indices_tensor, data_tensor, B, N, C, transpose_BC, alg = "GESPMM_ALG_DEFAULT"):
        if alg == "GESPMM_ALG_DEFAULT":
            alg = self.gespmmAlgSel(N, transpose_BC)
        
        if transpose_BC and (N <=32 and (N &(N-1)) == 0):
              pass
        else:
            if transpose_BC:
                match alg:
                    case "GESPMM_ALG_PARREDUCE_ROWBALANCE":
                        csrspmm.csrspmm_parreduce_rowbalance(M, K, nnz, indptr_tensor, indices_tensor, data_tensor, B, N, C)
                    case "GESPMM_ALG_PARREDUCE_NNZBALANCE":
                        csrspmm.csrspmm_parreduce_nnzbalance(M, K, nnz, indptr_tensor, indices_tensor, data_tensor, B, N, C)
                    case "GESPMM_ALG_SEQREDUCE_ROWBALANCE":
                        csrspmm.csrspmm_seqreduce_rowbalance(M, K, nnz, indptr_tensor, indices_tensor, data_tensor, B, N, C)
                    case "GESPMM_ALG_SEQREDUCE_NNZBALANCE":
                        csrspmm.csrspmm_seqreduce_nnzbalance(M, K, nnz, indptr_tensor, indices_tensor, data_tensor, B, N, C)
                    case "GESPMM_ALG_ROWCACHING_ROWBALANCE":
                        csrspmm.csrspmm_rowcaching_rowbalance(M, K, nnz, indptr_tensor, indices_tensor, data_tensor, B, N, C)
                    case "GESPMM_ALG_ROWCACHING_NNZBALANCE":
                        csrspmm.csrspmm_rowcaching_nnzbalance(M, K, nnz, indptr_tensor, indices_tensor, data_tensor, B, N, C)
                    case _:
                        print("Unknown algorithm")
            else:
                 match alg:
                    case "GESPMM_ALG_PARREDUCE_ROWBALANCE_NON_TRANSPOSE":
                        csrspmm.csrspmm_non_transpose_parreduce_rowbalance(M, K, nnz, indptr_tensor, indices_tensor, data_tensor, B, N, C)
                    case "GESPMM_ALG_PARREDUCE_NNZBALANCE_NON_TRANSPOSE":
                        csrspmm.csrspmm_non_transpose_parreduce_nnzbalance(M, K, nnz, indptr_tensor, indices_tensor, data_tensor, B, N, C)
                    case "GESPMM_ALG_SEQREDUCE_ROWBALANCE_NON_TRANSPOSE":
                        csrspmm.csrspmm_non_transpose_seqreduce_rowbalance(M, K, nnz, indptr_tensor, indices_tensor, data_tensor, B, N, C)
                    case "GESPMM_ALG_SEQREDUCE_NNZBALANCE_NON_TRANSPOSE":
                        csrspmm.csrspmm_non_transpose_seqreduce_nnzbalance(M, K, nnz, indptr_tensor, indices_tensor, data_tensor, B, N, C)
                    case _:
                        print("Unknown algorithm")
                      
                            
                            

                        

                          
            
            
        
            
                  
           