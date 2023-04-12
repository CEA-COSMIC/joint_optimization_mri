import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as open_file:
    install_requires = open_file.read()

setuptools.setup(
    name="jOpMRI",
    version="0.0.4",
    author="Chaithya G R",
    author_email="chaithyagr@gmail.com",
    description="Tools to benchmark and jointly optimize sampling and recon networks on the fastMRI dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chaithyagr/joint_optimization_mri",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    tests_require=['pytest>=5.0.1', 'pytest-cov>=2.7.1', 'pytest-pep8', 'pytest-runner', 'pytest-xdist'],
    python_requires='>=3.6',
)
