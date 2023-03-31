#csrspmm_seqreduce_nnzbalance
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='csrspmm_seqreduce_nnzbalance_ext',
    ext_modules=[
        CUDAExtension('csrspmm_seqreduce_nnzbalance_ext', [
                'csrspmm_seqreduce_nnzbalance.cpp',
                'kernel/csrspmm_seqreduce.cu',
        ]),
    ],
    cmdclass={
        'build_ext':BuildExtension
    })