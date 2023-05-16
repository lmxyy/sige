import copy
import glob
import os
import platform

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import build_ext, BuildExtension, CppExtension, CUDAExtension, IS_WINDOWS


def get_mps_extension(name, sources, *args, **kwargs):
    from torch.utils.cpp_extension import CppExtension

    extra_compile_args = {}
    extra_compile_args["cxx"] = ["-Wall", "-std=c++17"]
    extra_compile_args["cxx"] += ["-framework", "Metal", "-framework", "Foundation"]
    extra_compile_args["cxx"] += ["-ObjC++", "-O3"]

    kwargs["extra_compile_args"] = kwargs.get("extra_compile_args", {})
    kwargs["extra_compile_args"].update(extra_compile_args)
    return CppExtension(name, sources, *args, **kwargs)


def get_metal_extension(name, sources, *args, **kwargs):
    from setuptools import Extension

    for src in sources:
        assert os.path.splitext(src)[1] == ".metal", f"Expect .metal file, but get {src}."

    kwargs["language"] = "metal"
    return Extension(name, sources, *args, **kwargs)


class BuildMPSExtension(BuildExtension):
    def build_extensions(self) -> None:
        self._check_abi()

        for extension in self.extensions:
            if isinstance(extension.extra_compile_args, dict):
                for ext in ["cxx", "objc", "metal"]:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []
            self._add_compile_flag(extension, "-DTORCH_API_INCLUDE_EXTENSION_H")

            # See note [Pybind11 ABI constants]
            for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
                val = getattr(torch._C, f"_PYBIND11_{name}")
                if val is not None and not IS_WINDOWS:
                    self._add_compile_flag(extension, f'-DPYBIND11_{name}="{val}"')
            self._define_torch_extension_name(extension)
            self._add_gnu_cpp_abi_flag(extension)

        # register .mm .metal as valid type
        self.compiler.src_extensions += [".mm", ".metal"]
        original_compile = self.compiler._compile
        original_link = self.compiler.link
        original_object_filenames = self.compiler.object_filenames

        def darwin_wrap_single_compile(obj, src, ext, cc_args, extra_postargs, pp_opts) -> None:
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so

                if src.endswith(".metal"):
                    metal = ["xcrun", "metal"]
                    self.compiler.set_executable("compiler_so", metal)
                    if isinstance(cflags, dict):
                        cflags = cflags.get("metal", [])
                    else:
                        cflags = []
                elif isinstance(cflags, dict):
                    cflags = cflags["cxx"]

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                self.compiler.set_executable("compiler_so", original_compiler)

        def darwin_wrap_single_link(
            target_desc,
            objects,
            output_filename,
            output_dir=None,
            libraries=None,
            library_dirs=None,
            runtime_library_dirs=None,
            export_symbols=None,
            debug=0,
            extra_preargs=None,
            extra_postargs=None,
            build_temp=None,
            target_lang=None,
        ):
            if os.path.splitext(objects[0])[1].lower() == ".air":
                for obj in objects:
                    assert os.path.splitext(obj)[1].lower() == ".air", f"Expect .air file, but get {obj}."

                linker = ["xcrun", "metallib"]
                self.compiler.spawn(linker + objects + ["-o", output_filename])
            else:
                return original_link(
                    target_desc,
                    objects,
                    output_filename,
                    output_dir,
                    libraries,
                    library_dirs,
                    runtime_library_dirs,
                    export_symbols,
                    debug,
                    extra_preargs,
                    extra_postargs,
                    build_temp,
                    target_lang,
                )

        def darwin_wrap_object_filenames(source_filenames, strip_dir=0, output_dir=""):
            src_name = source_filenames[0]
            old_obj_extension = self.compiler.obj_extension
            if os.path.splitext(src_name)[1].lower() == ".metal":
                self.compiler.obj_extension = ".air"

            ret = original_object_filenames(source_filenames, strip_dir, output_dir)
            self.compiler.obj_extension = old_obj_extension

            return ret

        self.compiler._compile = darwin_wrap_single_compile
        self.compiler.link = darwin_wrap_single_link
        self.compiler.object_filenames = darwin_wrap_object_filenames
        build_ext.build_extensions(self)

    def get_ext_filename(self, ext_name):
        language = "cxx"
        for ext in self.extensions:
            if ext_name != ext.name:
                continue
            language = ext.language
            break

        if language == "metal":
            ext_path = ext_name.split(".")
            return os.path.join(*ext_path) + ".metallib"
        else:
            return super().get_ext_filename(ext_name)


if __name__ == "__main__":
    extra_compile_args = {"cxx": ["-g", "-O3", "-lgomp"], "nvcc": ["-O3"]}
    if platform.system() != "Darwin":
        extra_compile_args["cxx"].append("-fopenmp")
    build_extension_cls = BuildExtension

    cpu_extension = CppExtension(
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
        cuda_extension = CUDAExtension(
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

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        build_extension_cls = BuildMPSExtension
        # mps setting
        sources = glob.glob("sige/mps/*.cpp")
        sources += glob.glob("sige/mps/*.mm")
        sources += glob.glob("sige/*.cpp")
        name = "sige.mps"

        ext = get_mps_extension(name, sources)
        ext_modules.append(ext)

        # metal setting
        name = "metal_kernel"
        sources = glob.glob("sige/mps/*.metal")
        ext = get_metal_extension(name, sources)
        ext_modules.append(ext)
    with open("README.md", "r") as f:
        long_description = f.read()

    fp = open("sige/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    setup(
        name="sige",
        author="Muyang Li",
        author_email="muyangli@cs.cmu.edu",
        ext_modules=ext_modules,
        packages=find_packages(),
        cmdclass={"build_ext": build_extension_cls},
        install_requires=["torch>=1.7"],
        url="https://github.com/lmxyy/sige",
        description="Spatially Incremental Generative Engine (SIGE)",
        long_description=long_description,
        long_description_content_type="text/markdown",
        version=version,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
    )