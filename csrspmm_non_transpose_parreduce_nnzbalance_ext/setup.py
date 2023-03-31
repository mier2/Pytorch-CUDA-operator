from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

#csrspmm_non_transpose_parreduce_nnzbalance_ext
setup(
    name='csrspmm_non_transpose_parreduce_nnzbalance_ext',
    ext_modules=[
        CUDAExtension('csrspmm_non_transpose_parreduce_nnzbalance_ext', [
                'csrspmm_non_transpose_parreduce_nnzbalance.cpp',
                'kernel/csrspmm_non_transpose.cu',
        ]),
    ],
    cmdclass={
        'build_ext':BuildExtension
    })