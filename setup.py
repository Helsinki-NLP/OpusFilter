import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="opusfilter",
    version="0.1.0",
    author="Mikko Aulamo",
    author_email="mikko.aulamo@helsinki.fi",
    description="Tool for filtering parallel corpora",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Helsinki-NLP/OpusFilter",
    packages=setuptools.find_packages(),
    scripts=["bin/opus_filter"],
    install_requires=["opustools-pkg", "beautifulsoup4", "langid", "mosestokenizer", "pycld2", "PyYAML", "regex", "tqdm"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
