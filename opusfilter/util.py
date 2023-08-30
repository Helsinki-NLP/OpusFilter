"""Utility functions"""

import bz2
import gzip
import importlib
import io
import itertools
import json
import logging
import lzma
import os

import pandas as pd
from pandas import json_normalize
from tqdm import tqdm
import ruamel.yaml

from . import ConfigurationError


logger = logging.getLogger(__name__)


def lists_to_dicts(obj):
    """Convert lists in a JSON-style object to dicts recursively

    Examples:

    >>> lists_to_dicts([3, 4])
    {"0": 3, "1": 4}
    >>> lists_to_dicts([3, [4, 5]])
    {"0": 3, "1": {"0": 4, "1": 5}}
    >>> lists_to_dicts({"a": [3, 4], "b": []})
    {"a": {"0": 3, "1": 4}, "b": {}}

    """
    if isinstance(obj, dict):
        return {key: lists_to_dicts(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return {str(idx): lists_to_dicts(value) for idx, value in enumerate(obj)}
    return obj


def load_dataframe(data_file):
    """Load normalized scores dataframe from a JSON lines file"""
    data = []
    with file_open(data_file) as dfile:
        for line in dfile:
            try:
                data.append(lists_to_dicts(json.loads(line)))
            except json.decoder.JSONDecodeError as err:
                logger.error(line)
                raise err
    return pd.DataFrame(json_normalize(data))


def load_dataframe_in_chunks(data_file, chunksize):
    """Yield normalized scores dataframes from a chunked JSON lines file

    Use instead of load_dataframe if the data is too large to fit in memory.

    """
    with file_open(data_file) as dfile:
        for num, chunk in enumerate(grouper(dfile, chunksize)):
            data = []
            for line in chunk:
                try:
                    data.append(lists_to_dicts(json.loads(line)))
                except json.decoder.JSONDecodeError as err:
                    logger.error(line)
                    raise err
            logger.info("Processing chunk %s with %s lines", num, len(data))
            yield pd.DataFrame(json_normalize(data))


def import_class(config_dict, default_modules):
    """Import class from default modules or custom module defined in config

    The config_dict argument should have one key that corresponds to
    the class name, containing the arguments for the class, and
    optionally key "module" that gives the name of the custom module
    containing the class. If custom module is not provided, the class
    is searched from the default_modules, which should be a list of
    module objects.

    Returns the imported class name and the actual class.

    """
    custom_module = config_dict.pop('module') if 'module' in config_dict else None
    name = next(iter(config_dict.keys()))
    if custom_module:
        mod = importlib.import_module(custom_module)
        return name, getattr(mod, name)
    for module in default_modules:
        if hasattr(module, name):
            return name, getattr(module, name)
    raise KeyError(f'Class {name} not found in modules')


class FakeList:
    """Implements __getitem__ that returns always the same value"""

    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        return self.value


def check_args_compability(*args, required_types=None, choices=None, names=None):
    """Check that arguments of single value or list of values are compatible

    - Nth argument (plain value or within a list) must be instance of required_types[N] (if defined)
    - Nth argument (plain value or within a list) must be in of choices[N] (if defined)
    - If some arguments are lists, they have to be of the same length
    - If some arguments are lists, any non-list values are expanded to lists of the same length
    - If none of the arguments are lists, arguments are expanded to FakeList objects

    Return the input arguments expanded to lists or FakeLists when needed.

    """

    def type_error_msg(idx, type_, value):
        name = names[idx] if names else str(idx + 1)
        typestr = ' or '.join(t.__name__ for t in type_) if isinstance(type_, tuple) else type_.__name__
        return f"Values of argument '{name}' are not of the type {typestr}: {value}"

    def value_error_msg(idx, choices, value):
        name = names[idx] if names else str(idx + 1)
        return f"Values of argument '{name}' are not one of the allowed choices {choices}: {value}"

    def length_error_msg(idx, length, value):
        name = names[idx] if names else str(idx + 1)
        return f"List argument '{name}' do not match to the previous length {length}: {value}"

    def map_to_list(value, length):
        if length is None:
            return FakeList(value)
        if isinstance(value, list):
            return value
        return [value] * length

    list_len = None
    for idx, arg in enumerate(args):
        if isinstance(arg, list):
            if required_types and not all(isinstance(item, required_types[idx]) for item in arg):
                raise ConfigurationError(type_error_msg(idx, required_types[idx], arg))
            if choices and choices[idx] and not all((item in choices[idx]) for item in arg):
                raise ConfigurationError(value_error_msg(idx, choices[idx], arg))
            if list_len is None:
                list_len = len(arg)
            elif list_len != len(arg):
                raise ConfigurationError(length_error_msg(idx, list_len, arg))
        else:
            if required_types and not isinstance(arg, required_types[idx]):
                raise ConfigurationError(type_error_msg(idx, required_types[idx], arg))
            if choices and choices[idx] and arg not in choices[idx]:
                raise ConfigurationError(value_error_msg(idx, choices[idx], arg))
    if len(args) == 1:
        return map_to_list(args[0], list_len)
    return [map_to_list(arg, list_len) for arg in args]


def grouper(iterable, num):
    """Split data into fixed-length chunks"""
    iterable = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(iterable, num))
        if not chunk:
            return
        yield chunk


def file_open(filename, mode='r', encoding='utf8'):
    """Open file with implicit gzip/bz2 support

    Uses text mode by default regardless of the compression.

    In write mode, creates the output directory if it does not exist.

    """
    if 'w' in mode and os.path.dirname(filename) and not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    if filename.endswith('.bz2'):
        if mode in {'r', 'w', 'x', 'a'}:
            mode += 't'
        return bz2.open(filename, mode=mode, encoding=encoding)
    if filename.endswith('.xz'):
        if mode in {'r', 'w', 'x', 'a'}:
            mode += 't'
        return lzma.open(filename, mode=mode, encoding=encoding)
    if filename.endswith('.gz'):
        if mode in {'r', 'w', 'x', 'a'}:
            mode += 't'
        return gzip.open(filename, mode=mode, encoding=encoding)
    return open(filename, mode=mode, encoding=encoding)  # pylint: disable=R1732


def is_file_empty(filename):
    """Return whether compressed or plain file is empty"""
    with file_open(filename) as fobj:
        data = fobj.read(1)
    return not data


def file_download(url, localfile=None, chunk_size=None):
    """Download file from URL to a local file"""
    import requests
    if localfile is None:
        localfile = url.split('/')[-1]
    if chunk_size is None:
        chunk_size = 1024 * 1024
    with requests.get(url, stream=True) as req:
        req.raise_for_status()
        try:
            total_length = int(req.headers.get('content-length'))
        except (TypeError, ValueError):
            total_length = None
        pbar = tqdm(miniters=1, total=total_length, unit="B", unit_scale=True,
                    unit_divisor=1024, desc=localfile)
        with open(localfile, 'wb') as fobj:
            for chunk in req.iter_content(chunk_size=chunk_size):
                pbar.update(len(chunk))
                fobj.write(chunk)
                fobj.flush()
    return localfile


yaml = ruamel.yaml.YAML()


@ruamel.yaml.yaml_object(yaml)
class Var:
    """Reference for a variable"""
    yaml_tag = '!var'

    def __init__(self, value):
        self.value = value

    @classmethod
    def to_yaml(cls, representer, node):
        """Represent as YAML"""
        return representer.represent_scalar(cls.yaml_tag, f'{node.value}')

    @classmethod
    def from_yaml(cls, constructor, node):  # pylint: disable=W0613
        """Construct from YAML"""
        return cls(node.value)

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.value}')"

    def __str__(self):
        return self.__repr__()


@ruamel.yaml.yaml_object(yaml)
class VarStr(Var):
    """String template formatted using variables"""
    yaml_tag = '!varstr'


def yaml_dumps(obj):
    """Return a string containing YAML output from input object"""
    with io.StringIO() as iostream:
        yaml.dump(obj, iostream)
        iostream.seek(0)
        return iostream.read()


def count_lines(filename):
    """Count lines in a file"""
    with file_open(filename) as fobj:
        return sum(1 for _ in fobj)
