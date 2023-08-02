import sys
from glob import glob
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

if sys.version_info.major != 3:
    print(
        f"This Python is only compatible with Python 3, but you are running "
        f"Python {sys.version_info.major}. The installation will likely fail."
    )

ext_modules = [
    Pybind11Extension(
        "habitat_scanbot2d.astar_path_finder",
        sorted(glob("habitat_scanbot2d/src/*.cpp")),
        include_dirs=["habitat_scanbot2d/include", "/usr/include/eigen3"],
        extra_compile_args=["-g"],
    ),
]

setup(
    name="scanbot",
    version="0.2",
    description="A framework of robot autoscanning by using reinforcement learning.",
    author="Xi Xia, Hezhi Cao",
    author_email="againxx@mail.ustc.edu.cn, caohezhi21@mail.ustc.edu.cn",
    packages=find_packages(),
    package_data={
        "scanbot2d": ["py.typed"],
    },
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
)
