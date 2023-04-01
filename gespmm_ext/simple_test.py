import torch
import unittest
import scipy
import gespmm_ext as csrspmm
from data_loader import DataLoader

def fill_random(tensor):
        tensor.uniform_()
    


class TestSparseMM(unittest.TestCase):

    def test_simple(self):
        # Define new sparse input tensors and their corresponding CSR representation
        # Sparse Matrix A
        A_csr_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).cuda()
        A_csr_col_indices = torch.tensor([0, 2, 1, 0, 2], dtype=torch.int32).cuda()
        A_csr_row_ptrs = torch.tensor([0, 2, 3, 5], dtype=torch.int32).cuda()

        A_sparse = torch.tensor([[1.0, 0.0, 2.0],
                          [0.0, 3.0, 0.0],
                          [4.0, 0.0, 5.0]], dtype=torch.float32).cuda()

        # Dense Vector B
        B_dense = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).cuda()

        # Perform sparse matrix multiplication using torch.sparse.mm()
        C_expected = torch.sparse.mm(A_sparse, B_dense.view(3,1))


        # void csrspmm_non_transpose_parreduce_nnzbalance(int nrow, int ncol, int nnz,
        #                                         torch::Tensor &indptr, torch::Tensor &indices, torch::Tensor &data,
        #                                         const torch::Tensor &B, const int N, torch::Tensor &C){
        
        #seqreduce
        C_non_transpose_parreduce_nnzbalance = torch.zeros((3, 1), dtype=torch.float32).cuda()
        C_non_transpose_parreduce_rowbalance = torch.zeros((3, 1), dtype=torch.float32).cuda()
        C_non_transpose_seqreduce_nnzbalance = torch.zeros((3, 1), dtype=torch.float32).cuda()
        C_non_transpose_seqreduce_rowbalance = torch.zeros((3, 1), dtype=torch.float32).cuda()
        C_parreduce_nnzbalance = torch.zeros((3, 1), dtype=torch.float32).cuda()
        C_parreduce_rowbalance = torch.zeros((3, 1), dtype=torch.float32).cuda()
        C_rowcaching_nnzbalance = torch.zeros((3, 1), dtype=torch.float32).cuda()
        C_rowcaching_rowbalance = torch.zeros((3, 1), dtype=torch.float32).cuda()
        C_seqreduce_nnzbalance = torch.zeros((3, 1), dtype=torch.float32).cuda()
        C_seqreduce_rowbalance = torch.zeros((3, 1), dtype=torch.float32).cuda()
        
        csrspmm.csrspmm_non_transpose_parreduce_nnzbalance(3, 3, 5, A_csr_row_ptrs, A_csr_col_indices, A_csr_values, B_dense, 1, C_non_transpose_parreduce_nnzbalance)
        csrspmm.csrspmm_non_transpose_parreduce_rowbalance(3, 3, 5, A_csr_row_ptrs, A_csr_col_indices, A_csr_values, B_dense, 1, C_non_transpose_parreduce_rowbalance)
        csrspmm.csrspmm_non_transpose_seqreduce_nnzbalance(3, 3, 5, A_csr_row_ptrs, A_csr_col_indices, A_csr_values, B_dense, 1, C_non_transpose_seqreduce_nnzbalance)
        csrspmm.csrspmm_non_transpose_seqreduce_rowbalance(3, 3, 5, A_csr_row_ptrs, A_csr_col_indices, A_csr_values, B_dense, 1, C_non_transpose_seqreduce_rowbalance)
        csrspmm.csrspmm_parreduce_nnzbalance(3, 3, 5, A_csr_row_ptrs, A_csr_col_indices, A_csr_values, B_dense, 1, C_parreduce_nnzbalance)
        csrspmm.csrspmm_parreduce_rowbalance(3, 3, 5, A_csr_row_ptrs, A_csr_col_indices, A_csr_values, B_dense, 1, C_parreduce_rowbalance)
        csrspmm.csrspmm_rowcaching_nnzbalance(3, 3, 5, A_csr_row_ptrs, A_csr_col_indices, A_csr_values, B_dense, 1, C_rowcaching_nnzbalance)
        csrspmm.csrspmm_rowcaching_rowbalance(3, 3, 5, A_csr_row_ptrs, A_csr_col_indices, A_csr_values, B_dense, 1, C_rowcaching_rowbalance)
        csrspmm.csrspmm_seqreduce_nnzbalance(3, 3, 5, A_csr_row_ptrs, A_csr_col_indices, A_csr_values, B_dense, 1, C_seqreduce_nnzbalance)
        csrspmm.csrspmm_seqreduce_rowbalance(3, 3, 5, A_csr_row_ptrs, A_csr_col_indices, A_csr_values, B_dense, 1, C_seqreduce_rowbalance)

        # Check if the output is correct
        self.assertTrue(torch.allclose(C_parreduce_nnzbalance, C_expected, atol=1e-7), "Simple Test for csrspmm_non_transpose_parreduce_nnzbalance Failed")
        self.assertTrue(torch.allclose(C_non_transpose_parreduce_rowbalance, C_expected, atol=1e-7), "Simple Test for csrspmm_non_transpose_parreduce_rowbalance Failed")
        self.assertTrue(torch.allclose(C_non_transpose_seqreduce_nnzbalance, C_expected, atol=1e-7), "Simple Test for csrspmm_non_transpose_seqreduce_nnzbalance Failed")
        self.assertTrue(torch.allclose(C_non_transpose_seqreduce_rowbalance, C_expected, atol=1e-7), "Simple Test for csrspmm_non_transpose_seqreduce_rowbalance Failed")
        self.assertTrue(torch.allclose(C_parreduce_nnzbalance, C_expected, atol=1e-7), "Simple Test for csrspmm_parreduce_nnzbalance Failed")
        self.assertTrue(torch.allclose(C_parreduce_rowbalance, C_expected, atol=1e-7), "Simple Test for csrspmm_parreduce_rowbalance Failed")
        self.assertTrue(torch.allclose(C_rowcaching_nnzbalance, C_expected, atol=1e-7), "Simple Test for csrspmm_rowcaching_nnzbalance Failed")
        self.assertTrue(torch.allclose(C_rowcaching_rowbalance, C_expected, atol=1e-7), "Simple Test for csrspmm_rowcaching_rowbalance Failed")
        self.assertTrue(torch.allclose(C_seqreduce_nnzbalance, C_expected, atol=1e-7), "Simple Test for csrspmm_seqreduce_nnzbalance Failed")
        self.assertTrue(torch.allclose(C_seqreduce_rowbalance, C_expected, atol=1e-7), "Simple Test for csrspmm_seqreduce_rowbalance Failed")



if __name__ == '__main__':
    unittest.main()