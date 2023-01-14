from os import path
from io import open
from setuptools import setup, find_packages


here = path.abspath(path.dirname(__file__))

## get __version__ from version.py
__version__ = None
ver_file = path.join("calotron", "version.py")
with open(ver_file) as f:
  exec(f.read())

## load README
def readme():
  readme_path = path.join(here, "README.md")
  with open(readme_path, encoding="utf-8") as f:
    return f.read()

## load requirements
def requirements():
  requirements_path = path.join(here, "requirements.txt")
  with open(requirements_path, encoding="utf-8") as f:
    return f.read().splitlines()

setup (
        name = "calotron",
        version = __version__,
        description  = "Transformer-based model to fast-simulate the LHCb ECAL detector",
        long_description = readme(),
        long_description_content_type = "text/markdown",
        url = "https://github.com/mbarbetti/calotron",
        author = "Matteo Barbetti",
        author_email = "matteo.barbetti@fi.infn.it",
        maintainer = "Matteo Barbetti",
        maintainer_email = "matteo.barbetti@fi.infn.it",
        license = "MIT",
        keywords = ["lhcb-experiment", "ultrafast-simulation", "lamarr",
                    "calorimeter", "machine-learning", "transformer"],
        packages = find_packages(),
        package_data = {},
        include_package_data = True,
        install_requires = requirements(),
        python_requires  = ">=3.7, <=3.10",
        classifiers = [
                        "Development Status :: 3 - Alpha",
                        "Intended Audience :: Education",
                        "Intended Audience :: Developers",
                        "Intended Audience :: Science/Research",
                        "License :: OSI Approved :: MIT License",
                        "Programming Language :: Python :: 3 :: Only",
                        "Programming Language :: Python :: 3.7",
                        "Programming Language :: Python :: 3.8",
                        "Programming Language :: Python :: 3.9",
                        "Programming Language :: Python :: 3.10",
                        "Topic :: Scientific/Engineering",
                        "Topic :: Scientific/Engineering :: Mathematics",
                        "Topic :: Scientific/Engineering :: Artificial Intelligence",
                        "Topic :: Software Development",
                        "Topic :: Software Development :: Libraries",
                        "Topic :: Software Development :: Libraries :: Python Modules",
                      ],
  )