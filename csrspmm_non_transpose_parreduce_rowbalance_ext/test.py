import torch
import numpy
import unittest
from torch.autograd import gradcheck
import csrspmm_non_transpose_parreduce_rowbalance_ext as csrspmm



class TestCSRSPMM(unittest.TestCase):
      def test_csrspmm(self):
        # Define the input tensors and their corresponding CSR representation
        A_dense = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]], dtype=torch.float32).cuda()
        B = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32).cuda()

        # The CSR representation of A_dense
        csr_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32).cuda()
        csr_col_indices = torch.tensor([0, 2, 1, 0, 2], dtype=torch.int32).cuda()
        csr_row_ptrs = torch.tensor([0, 2, 3, 5], dtype=torch.int32).cuda()

        # Call the PyTorch extension function
        C = torch.zeros((3, 2), dtype=torch.float32).cuda()
        # void csrspmm_non_transpose_parreduce_rowbalance(int nrow, int ncol, int nnz,
        #                                         torch::Tensor &indptr, torch::Tensor &indices, torch::Tensor &data,
        #                                         const torch::Tensor &B, const int N, torch::Tensor &C)
        csrspmm.csrspmm_non_transpose_parreduce_rowbalance(3, 3, 5, csr_row_ptrs, csr_col_indices, csr_values, B,2, C)
        # Compute the expected result
        C_expected = torch.matmul(A_dense, B)
        print("C_expected:")
        print(C_expected)
        
        print("C")
        print(C)


        # Check if the output is correct (use a small tolerance for floating-point errors)
        self.assertTrue(torch.allclose(C, C_expected, atol=1e-7))
        

if __name__ == '__main__':
    unittest.main()