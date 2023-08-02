from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="channels_constructor",
    ext_modules=[
        CUDAExtension('channels_constructor', [
            "channels_constructor.cpp",
            "channels_constructor_kernel.cu",
        ])
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
