import os, sys, setuptools
from setuptools import setup


def read(fname):
    path = os.path.join(os.path.dirname(__file__), fname)
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    return data


def process_requirements(fname):
  """Modify list of requirements to support local bleurt install.

  The format for local imports when using requirements.txt for a direct install
  and the foramt needed for setup.py are different. This function translates
  between the two.
  """
  requirements = read(fname)

  def _process_local(x):
    if "bleurt" in  x:
      bleurt_path = os.path.abspath(x)
      return f"bleurt @ file://localhost/{bleurt_path}#egg=bleurt"
    else:
      return x

  processed_requirements = [
      _process_local(x) for x in requirements.strip().split('\n')]
  return processed_requirements


setup(
    name="colar",
    version="0.0.1",
    author="Miguel Bragança",
    author_email="mp1820@ic.ac.uk",
    description=(
        "A framework for Compression Of LArge language models for Robotics." 
    ),
    # include PEP-420 subpackages under benchmark_tasks
    packages=(
        setuptools.find_packages()
    ),
    long_description=read("README.md"),
    long_description_content_type='text/markdown',
    install_requires=process_requirements("requirements.txt"),
    package_data={
        "": [
            "*.json",
            "*.jsonl",
            "*.txt",
            "*.tsv",
            "*.csv",
            "*.npz",
            "*.ckpt",
            "*.gz",
            "*.zip",
            "*.yaml",
            "*.pkl",
        ]
    },
)