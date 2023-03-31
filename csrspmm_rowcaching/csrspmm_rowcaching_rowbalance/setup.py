#csrspmm_rowcaching_rowbalance
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='csrspmm_rowcaching_rowbalance_ext',
    ext_modules=[
        CUDAExtension('csrspmm_rowcaching_rowbalance_ext', [
                'csrspmm_rowcaching_rowbalance.cpp',
                'kernel/csrspmm_rowcaching.cu',
        ]),
    ],
    cmdclass={
        'build_ext':BuildExtension
    })