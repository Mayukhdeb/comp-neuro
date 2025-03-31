import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="code/material for the computational neuroscience course @ Georgia Tech",
    version="0.0.0",
    author="Mayukh Deb, N. Apurva Ratan Murty",
    author_email="mayukh@gatech.edu",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mayukhdeb/comp-neuro",
    packages=setuptools.find_packages(),
    install_requires=None,
)