#csrspmm_seqreduce_rowbalance
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='csrspmm_seqreduce_rowbalance_ext',
    ext_modules=[
        CUDAExtension('csrspmm_seqreduce_rowbalance_ext', [
                'csrspmm_seqreduce_rowbalance.cpp',
                'kernel/csrspmm_seqreduce.cu',
        ]),
    ],
    cmdclass={
        'build_ext':BuildExtension
    })