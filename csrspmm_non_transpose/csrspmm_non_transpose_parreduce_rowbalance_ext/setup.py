from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='csrspmm_non_transpose_parreduce_rowbalance_ext',
    ext_modules=[
        CUDAExtension('csrspmm_non_transpose_parreduce_rowbalance_ext', [
                'csrspmm_non_transpose_parreduce_rowbalance.cpp',
                'kernel/csrspmm_non_transpose.cu',
        ]),
    ],
    cmdclass={
        'build_ext':BuildExtension
    })

