import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Saber",
    version="0.0.1",
    author="John Giorgi",
    author_email="johnmgiorgi@gmail.com",
    license="MIT",
    description="Saber: Sequence Annotator for Biomedical Entities and Relations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BaderLab/Saber",
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    classifiers=(
        "Development Status :: 3 - Alpha",
        "Framework :: Flask",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6"
        "Operating System :: OS Independent",
    ),
    install_requires=[
        'tensorflow>=1.9.0',
        'Flask>=1.0.2',
        'gensim>=3.4.0'
        'keras==2.2.0',
        'PTable>=0.9.2',
        'scikit-learn>=0.19.1',
        'spacy>=2.0.11',
    ],
    include_package_data=True,
)
