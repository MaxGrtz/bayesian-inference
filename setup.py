import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bayes-vi",
    version="0.2.0",
    author="Maximilian Gartz",
    author_email="maximilian.gartz@outlook.de",
    description="A package for bayesian inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MaxGrtz/bayesian-inference",
    packages=setuptools.find_packages(),
    install_requires=[
        'decorator>=4.4.2',
        'numpy>=1.20.1',
        'tensorflow>=2.4.1',
        'pandas>=1.2.3',
        'fastprogress>=1.0.0',
        'tensorflow_probability>=0.12.1'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)