from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'mcts_core',
        ['mcts_core.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['/O2', '/fp:fast']  # MSVC optimizations
    ),
]

setup(
    name='mcts_core',
    ext_modules=ext_modules,
)
