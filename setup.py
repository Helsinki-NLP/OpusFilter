import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "opustools",
    "beautifulsoup4",
    "fasttext",
    "graphviz",
    "langid",
    "matplotlib",
    "fast-mosestokenizer",
    "pandas>=1.0.0",
    "pycld2",
    "pyhash",
    "sentence-splitter",
    "ruamel.yaml>=0.15.0",
    "regex",
    "requests",
    "scikit-learn",
    "tqdm"
]

jieba_require = [
    'jieba>=0.42'
]

tests_require = [
    'pytest'
]

all_require = jieba_require + tests_require

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
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require={'test': tests_require, 'jieba': jieba_require, 'all': all_require},
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    python_requires=">=3.6",
)
