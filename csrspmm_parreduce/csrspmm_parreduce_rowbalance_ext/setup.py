#csrspmm_parreduce_rowbalance
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='csrspmm_parreduce_rowbalance_ext',
    ext_modules=[
        CUDAExtension('csrspmm_parreduce_rowbalance_ext', [
                'csrspmm_parreduce_rowbalance.cpp',
                'kernel/csrspmm_parreduce.cu',
        ]),
    ],
    cmdclass={
        'build_ext':BuildExtension
    })

