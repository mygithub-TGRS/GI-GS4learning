from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization_wfeat_optimized",
    packages=["diff_gaussian_rasterization_wfeat_optimized"],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_wfeat_optimized._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                "nvcc": [
                    "-Xcompiler",
                    "-fno-gnu-unique",
                    "-I" + os.path.join(_THIS_DIR, "third_party/glm/"),
                ]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
