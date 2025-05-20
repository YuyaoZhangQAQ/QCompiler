from setuptools import setup, find_packages

with open("requirements.txt", encoding = "utf-8") as fp:
    requirements = fp.read().splitlines()

setup(
    name = "QCompiler",
    version = "0.1.0",
    packages = find_packages(),
    install_requires = requirements,
    include_package_data = True,
    python_requires = ">=3.10",
)