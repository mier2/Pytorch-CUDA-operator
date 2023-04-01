from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='gespmm_ext',
    ext_modules=[
        CUDAExtension('gespmm_ext', [
                'gespmm_wrapper.cpp',
                'kernel/csrspmm_non_transpose.cu',
                'kernel/csrspmm_parreduce.cu',
                'kernel/csrspmm_rowcaching.cu',
                'kernel/csrspmm_seqreduce.cu',

        ]),
    ],
    cmdclass={
        'build_ext':BuildExtension
    })