import torch
import numpy
import unittest
import csrspmm_parreduce_rowbalance_ext as csrspmm

class TestSparseMM(unittest.TestCase):
    def test_sparse_mm(self):
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
        C_expected = torch.sparse.mm(A_sparse, B_dense.view(3, 1))

        # void csrspmm_non_transpose_parreduce_nnzbalance(int nrow, int ncol, int nnz,
        #                                         torch::Tensor &indptr, torch::Tensor &indices, torch::Tensor &data,
        #                                         const torch::Tensor &B, const int N, torch::Tensor &C){
        C = torch.zeros((3, 1), dtype=torch.float32).cuda()
        csrspmm.csrspmm_parreduce_rowbalance(3, 3, 5, A_csr_row_ptrs, A_csr_col_indices, A_csr_values, B_dense, 1, C)

        print("C_expected:")
        print(C_expected)

        print("C")
        print(C)

        # Check if the output is correct
        self.assertTrue(torch.allclose(C, C_expected, atol=1e-7))

if __name__ == '__main__':
    unittest.main()