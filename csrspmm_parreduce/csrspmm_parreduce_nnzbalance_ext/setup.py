#csrspmm_parreduce_nnzbalance
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='csrspmm_parreduce_nnzbalance_ext',
    ext_modules=[
        CUDAExtension('csrspmm_parreduce_nnzbalance_ext', [
                'csrspmm_parreduce_nnzbalance.cpp',
                'kernel/csrspmm_parreduce.cu',
        ]),
    ],
    cmdclass={
        'build_ext':BuildExtension
    })

