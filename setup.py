import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Saber",
    version="0.0.1",
    author="John Giorgi",
    author_email="johnmgiorgi@gmail.com",
    description="Saber: Sequence Annotator for Biomedical Entities and Relations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BaderLab/Saber",
    python_requires='>=3.5',
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        'flask>=1.0.2',
        'keras>=2.2.0',
        'PTable',
        'scikit-learn>=0.19.1',
        'spacy>=2.0.11',
        'tensorflow>=1.9.0',
    ],
)
