import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="opusfilter",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    author="Mikko Aulamo, Sami Virpioja",
    author_email="mikko.aulamo@helsinki.fi",
    description="Toolbox for filtering parallel corpora",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Helsinki-NLP/OpusFilter",
    packages=setuptools.find_packages(),
    scripts=["bin/opusfilter", "bin/opusfilter-cmd", "bin/opusfilter-diagram", "bin/opusfilter-duplicates", "bin/opusfilter-scores", "bin/opusfilter-test"],
    install_requires=["opustools", "beautifulsoup4", "graphviz", "langid", "matplotlib", "fast-mosestokenizer", "pandas>=0.24.0", "pycld2", "pyhash", "PyYAML", "regex", "scikit-learn", "tqdm"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    python_requires=">=3.6",
)
