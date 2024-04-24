from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ProSTGrid',
    ext_modules=[
        CUDAExtension('ProSTGrid', [
            'ProSTGrid.cpp',
            'ProSTGrid_kernel.cu',
        ],
        extra_compile_args={"nvcc": ["-ccbin=/usr/local/cuda/bin/gcc"]}
                      ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
