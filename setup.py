from distutils.core import setup
import clumpy

packages = ['clumpy']
install_requires = ['numpy>=1.8']
ext_modules = []

setup(
    name='clumpy',
    author="Brian Kimmig",
    author_email='brian.kimmig@gmail.com',
    url="https://github.com/bkimmig/clumpy",
    license="",
    description="Python Expectation Maximization for Astronomy",
    long_description=open("README.md").read(),
    classifiers=["Development Status :: 3 - Alpha",
                 "Intended Audience :: Developers",
                 "Intended Audience :: Science/Research",
                 "License :: OSI Approved :: BSD License",
                 "Natural Language :: English",
                 "Programming Language :: Python",
                 "Topic :: Scientific/Engineering :: Mathematics",
                 "Topic :: Scientific/Engineering :: Physics",
                 "Topic :: Scientific/Engineering :: Astronomy"],
    platforms='any',
    # version=clumpy.__version__,
    packages=packages,
    ext_modules=ext_modules,
    install_requires=install_requires,
)
