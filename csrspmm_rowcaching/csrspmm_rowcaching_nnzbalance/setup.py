#csrspmm_rowcaching_nnzbalance
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='csrspmm_rowcaching_nnzbalance_ext',
    ext_modules=[
        CUDAExtension('csrspmm_rowcaching_nnzbalance_ext', [
                'csrspmm_rowcaching_nnzbalance.cpp',
                'kernel/csrspmm_rowcaching.cu',
        ]),
    ],
    cmdclass={
        'build_ext':BuildExtension
    })