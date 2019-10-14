import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="saber",
    version="0.1.0-alpha",
    author="John Giorgi",
    author_email="johnmgiorgi@gmail.com",
    license="MIT",
    description="Saber: Sequence Annotator for Biomedical Entities and Relations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BaderLab/saber",
    python_requires='>=3.5',
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Framework :: Flask",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    keywords=[
        'Natural Language Processing',
        'Named Entity Recognition',
    ],
    install_requires=[
        'scikit-learn>=0.20.1',
        'tensorflow>=1.12.0',
        'Flask>=1.0.2',
        'waitress>=1.1.0',
        'keras==2.2.4',
        'PTable>=0.9.2',
        'spacy>=2.0.11, <=2.0.13',
        'gensim>=3.4.0',
        'nltk>=3.3',
        'googledrivedownloader>=0.3',
        'google-compute-engine',
        'msgpack==0.5.6',
        'keras-contrib @ git+https://www.github.com/keras-team/keras-contrib.git',
        'en-coref-md @ https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_md-3.0.0/en_coref_md-3.0.0.tar.gz',
    ],
    include_package_data=True,
    # allows us to install + run tests with `python setup.py test`
    # https://docs.pytest.org/en/latest/goodpractices.html#integrating-with-setuptools-python-setup-py-test-pytest-runner
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    zip_safe=False,
)
