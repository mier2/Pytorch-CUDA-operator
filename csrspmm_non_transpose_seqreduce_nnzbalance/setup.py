from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='csrspmm_non_transpose_seqreduce_nnzbalance_ext',
    ext_modules=[
        CUDAExtension('csrspmm_non_transpose_seqreduce_nnzbalance_ext', [
                'csrspmm_non_transpose_seqreduce_nnzbalance.cpp',
                'kernel/csrspmm_non_transpose.cu',
        ]),
    ],
    cmdclass={
        'build_ext':BuildExtension
    })


