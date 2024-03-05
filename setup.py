import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="snnk",
    version="1.0.0",
    author="SNNK Authors",
    author_email="arijit.sehanobish1@gmail.com",
    description="Package for Scalable Neural Network Kernels",
    long_description="TODO",
    long_description_content_type="text/markdown",
    url="https://github.com/neural-network-kernels",
    packages=['nnk', 'text_classification', 'vision'],
    package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
