from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
version_dict = {}
with open(Path(__file__).parents[0] / "source/_version.py") as fp:
    exec(fp.read(), version_dict)
version = version_dict["__version__"]
del version_dict

setup(
    name="pretrain_repo",
    version=version,
    author="mn",
    description="pretrain_repo",
    python_requires=">=3.8",
    packages=find_packages(include=["pretrain_repo", "pretrain_repo.*"]),
    zip_safe=True,
)
