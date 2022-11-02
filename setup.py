import platform

import torch
from setuptools import setup
from torch.utils import cpp_extension

if __name__ == "__main__":
    extra_compile_args = {"cxx": ["-g", "-O3", "-lgomp"], "nvcc": ["-O3"]}
    if platform.system() != "Darwin":
        extra_compile_args["cxx"].append("-fopenmp")

    cpu_extension = cpp_extension.CppExtension(
        name="sige.cpu",
        sources=[
            "sige/cpu/gather.cpp",
            "sige/cpu/scatter.cpp",
            "sige/cpu/scatter_gather.cpp",
            "sige/cpu/common_cpu.cpp",
            "sige/cpu/pybind_cpu.cpp",
            "sige/common.cpp",
        ],
        extra_compile_args=extra_compile_args,
    )
    ext_modules = [cpu_extension]
    if torch.cuda.is_available():
        cuda_extension = cpp_extension.CUDAExtension(
            name="sige.cuda",
            sources=[
                "sige/cuda/gather.cpp",
                "sige/cuda/gather_kernel.cu",
                "sige/cuda/scatter.cpp",
                "sige/cuda/scatter_kernel.cu",
                "sige/cuda/scatter_gather.cpp",
                "sige/cuda/scatter_gather_kernel.cu",
                "sige/cuda/common_cuda.cu",
                "sige/cuda/pybind_cuda.cpp",
                "sige/common.cpp",
            ],
            extra_compile_args=extra_compile_args,
        )
        ext_modules.append(cuda_extension)

    with open("README.md", "r") as fh:
        long_description = fh.read()

    setup(
        name="sige",
        author="Muyang Li",
        author_email="muyangli@cs.cmu.edu",
        ext_modules=ext_modules,
        packages=["sige"],
        cmdclass={"build_ext": cpp_extension.BuildExtension},
        install_requires=["torch>=1.7"],
        url="https://github.com/lmxyy/sige",
        description="Spatially Incremental Generative Engine (SIGE)",
        long_description=long_description,
        long_description_content_type="text/markdown",
        version="0.1.3",
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
    )
