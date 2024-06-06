from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="euclidDistance",
    ext_modules=[
        CUDAExtension(
            name="euclidDistance",
            sources=["euclidDistance.cu", "ext.cpp"]
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)