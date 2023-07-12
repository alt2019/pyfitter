from setuptools import setup#, Extension
from distutils.core import Extension
from Cython.Build import cythonize



ext_modules = [
    Extension('neutrino_oscillations', sources=['oscillation.pyx']),
    Extension('fitter', sources=['fit.pyx']),
]

setup(
    name='fitter',
    # ext_modules=cythonize(extensions),
    ext_modules=cythonize(ext_modules),
)
