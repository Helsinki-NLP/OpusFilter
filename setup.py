import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "setuptools",
    "opustools>=1.6.2",
    "beautifulsoup4>=4.8.0",
    "graphviz",
    "py3langid>=0.2.2",
    "matplotlib",
    "morfessor",
    "opus-fast-mosestokenizer>=0.0.8.5",
    "pandas>=1.0.0",
    "xxhash>=3.2.0",
    "sentence-splitter",
    "rapidfuzz",
    "ruamel.yaml>=0.15.0",
    "regex",
    "requests",
    "scikit-learn",
    "subword_nmt",
    "tqdm",
    "lingua-language-detector>=1.3.0"
]

pycld2_require = [
    "pycld2"
]

fasttext_require = [
    "py3langid<0.3.0",  # 0.3.0 requires numpy 2.0.0
    "numpy<2.0.0",
    "fasttext"
]

eflomal_require = [
    'eflomal>=2.0.0'
]

jieba_require = [
    'jieba>=0.42'
]

mecab_require = [
    'mecab-python3>=1.0.8',
    'unidic-lite'
]

laser_require = [
    'laserembeddings'
]

varikn_require = [
    'varikn'
]

tests_require = [
    'pytest'
]

docs_require = [
    'myst-parser',
    'sphinx',
    'sphinx-rtd-theme',
    'sphinxcontrib-bibtex'
]

all_require = pycld2_require + fasttext_require + eflomal_require + jieba_require + \
    mecab_require + laser_require + varikn_require + tests_require + docs_require

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
    scripts=["bin/opusfilter", "bin/opusfilter-autogen",
        "bin/opusfilter-cmd", "bin/opusfilter-diagram", "bin/opusfilter-duplicates",
        "bin/opusfilter-scores", "bin/opusfilter-test"],
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require={'test': tests_require, 'pycld2': pycld2_require, 'fasttext': fasttext_require,
                    'eflomal': eflomal_require, 'jieba': jieba_require, 'mecab': mecab_require,
                    'laser': laser_require, 'varikn': varikn_require, 'docs': docs_require,
                    'all': all_require},
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    python_requires=">=3.8",
)
