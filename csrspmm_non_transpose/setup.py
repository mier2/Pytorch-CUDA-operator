from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='csrspmm_non_transpose',
    ext_modules=[
        CUDAExtension('non_transpose_cuda', [
                'non_transpose_cuda.cpp',
                'csrspmm_non_transpose.cu',
        ]),
    ],
    cmdclass={
        'build_ext':BuildExtension
    })