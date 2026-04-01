"""Utility functions"""

import bz2
import copy
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

TRAINING_STEP_TYPES = {'train_ngram', 'train_alignment', 'train_bpe', 'train_spm'}


def get_inputs(step):
    """Return inputs of the step.

    Includes explicit inputs from parameters plus implicit inputs from
    depends_on field (for files required by the step but not in standard I/O).
    """
    params = step.get('parameters', {})
    inputs = params.get('inputs', [])
    if inputs and isinstance(inputs[0], list):
        inputs = [item for sublist in inputs for item in sublist]
    for single_input in ['input', 'src_input', 'tgt_input']:
        if single_input in params:
            inputs.append(params[single_input])
    step_type = step.get('type', '')
    if step_type in TRAINING_STEP_TYPES:
        for data_input in ['data', 'src_data', 'tgt_data']:
            if data_input in params:
                inputs.append(params[data_input])
    # Add implicit inputs from depends_on field
    depends_on = step.get('depends_on', [])
    if isinstance(depends_on, str):
        depends_on = [depends_on]
    inputs.extend(depends_on)
    return inputs


def convert_vars_to_strings(obj):
    """Recursively convert Var/VarStr objects to their string values."""
    if isinstance(obj, Var):
        return obj.value
    if isinstance(obj, dict):
        return {key: convert_vars_to_strings(val) for key, val in obj.items()}
    if isinstance(obj, list):
        return [convert_vars_to_strings(item) for item in obj]
    return obj


def get_outputs(step):
    """Return output filenames for a step."""
    params = step.get('parameters', {})
    outputs = params.get('outputs', [])
    if isinstance(outputs, str):
        outputs = [outputs]
    for single_output in ['output', 'src_output', 'tgt_output']:
        if single_output in params:
            outputs.append(params[single_output])
    step_type = step.get('type', '')
    if step_type in TRAINING_STEP_TYPES and 'model' in params:
        outputs.append(params['model'])
    return outputs


def get_other_params(step):
    """Return parameters of the step excluding i/o."""
    params = copy.copy(step.get('parameters', {}))
    for to_remove in ['input', 'inputs', 'output', 'outputs', 'src_output', 'tgt_output']:
        if to_remove in params:
            del params[to_remove]
    step_type = step.get('type', '')
    if step_type in TRAINING_STEP_TYPES:
        for to_remove in ['data', 'src_data', 'tgt_data']:
            if to_remove in params:
                del params[to_remove]
    return params


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


class _JsonlTextReader:
    """Wraps a readable file object to transparently parse JSONL.

    Each line is parsed as JSON; if the result is a string it is used
    directly, otherwise the *text* key is extracted.  This lets both
    ``json.dumps(text)`` and ``{"text": text}`` formats round-trip
    correctly.
    """

    def __init__(self, fobj):
        self._fobj = fobj

    def readline(self):
        line = self._fobj.readline()
        if not line:
            return line
        obj = json.loads(line.rstrip('\n'))
        text = obj if isinstance(obj, str) else obj['text']
        return text + '\n'

    def __iter__(self):
        return self

    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def close(self):
        self._fobj.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class _JsonlTextWriter:
    """Wraps a writable file object to transparently write JSONL.

    Each call to ``write(text)`` serialises *text* as a JSON string
    and appends ``\\n``, producing one valid JSON line per record.

    A trailing ``\\n`` in *text* (the record-separator convention
    used throughout the pipeline) is stripped before serialisation
    so that it is not embedded in the JSON value.
    """

    def __init__(self, fobj):
        self._fobj = fobj

    def write(self, text):
        if text.endswith('\n'):
            text = text[:-1]
        self._fobj.write(json.dumps(text, ensure_ascii=False) + '\n')

    def close(self):
        self._fobj.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class _PlainTextWriter:
    """Wraps a writable file object to warn on embedded newlines.

    Writing text with embedded newlines to a non-JSONL file inflates
    line-based counts, which can cause misalignment in downstream
    steps that rely on ``wc -l`` or equivalent.  A warning is issued
    once per file when the first embedded newline is detected.
    """

    def __init__(self, fobj, filename):
        self._fobj = fobj
        self._filename = filename
        self._newline_warned = False

    def write(self, text):
        if not self._newline_warned:
            content = text[:-1] if text.endswith('\n') else text
            if '\n' in content:
                logger.warning(
                    "Text written to %s contains embedded newlines. "
                    "This inflates line-based counts and may cause "
                    "misalignment in downstream steps. Use a .jsonl "
                    "file to preserve multi-line segments.",
                    self._filename)
                self._newline_warned = True
        self._fobj.write(text)

    def flush(self):
        self._fobj.flush()

    def close(self):
        self._fobj.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _strip_compression_suffix(filename):
    """Remove a recognised compression suffix from *filename*."""
    for ext in ('.gz', '.bz2', '.xz'):
        if filename.endswith(ext):
            return filename[:-len(ext)]
    return filename


def text_file_open(filename, mode='r', encoding='utf8'):
    """Open a text file with transparent JSONL support.

    When *filename* ends with ``.jsonl`` (possibly followed by a
    compression suffix such as ``.gz``, ``.bz2`` or ``.xz``):

    * ``'r'`` — each ``readline()`` / iteration returns the
      deserialised text (``json.dumps(text)`` or ``{"text": text}``
      are both accepted).
    * ``'w'`` / ``'a'`` / ``'x'`` — each ``write(text)`` serialises
      *text* via ``json.dumps`` and writes one JSON line.

    For any other extension the behaviour is identical to
    :func:`file_open`.
    """
    is_jsonl = _strip_compression_suffix(filename).endswith('.jsonl')
    fobj = file_open(filename, mode=mode, encoding=encoding)
    if not is_jsonl:
        if 'w' in mode or 'a' in mode or 'x' in mode:
            return _PlainTextWriter(fobj, filename)
        return fobj
    if 'r' in mode:
        return _JsonlTextReader(fobj)
    if 'w' in mode or 'a' in mode or 'x' in mode:
        return _JsonlTextWriter(fobj)
    return fobj


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

    def __fspath__(self):
        return self.value


@ruamel.yaml.yaml_object(yaml)
class VarStr(Var):
    """String template formatted using variables"""
    yaml_tag = '!varstr'

    def __str__(self):
        return self.value


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


def expand_step_parameters(obj, namespace):
    """Expand Var and VarStr objects in obj using namespace."""
    if isinstance(obj, list):
        return [expand_step_parameters(x, namespace) for x in obj]
    if isinstance(obj, dict):
        return {expand_step_parameters(key, namespace): expand_step_parameters(value, namespace)
                for key, value in obj.items()}
    if isinstance(obj, VarStr):
        try:
            return obj.value.format(**namespace)
        except (KeyError, IndexError):
            return obj.value
    if isinstance(obj, Var):
        return namespace.get(obj.value, obj.value)
    return obj


def expand_single_step(step, substep_index, common_constants=None):
    """Expand a single step for a specific substep index.

    Args:
        step: Step configuration dict
        substep_index: Which substep to expand (0-indexed)
        common_constants: Dictionary of common constants from config

    Returns:
        Step config with variables expanded for the given substep.
        Sets 'parameters' to expanded values and clears 'variables'.
    """
    common_constants = common_constants or {}
    variables = step.get('variables', {})

    if not variables:
        namespace = copy.copy(common_constants)
        namespace.update(step.get('constants', {}))
        expanded_params = expand_step_parameters(step.get('parameters', {}), namespace)
        result = copy.deepcopy(step)
        result['parameters'] = expanded_params
        result['variables'] = {}
        return result

    lengths = set()
    for key, value in variables.items():
        if not isinstance(value, list):
            raise ConfigurationError(f"Variable {key} does not define a list")
        lengths.add(len(value))
    if len(lengths) > 1:
        raise ConfigurationError(
            f"Variables have inconsistent lengths: {lengths}. "
            "All variables must have the same number of values."
        )
    num_choices = list(lengths)[0] if lengths else 0

    if substep_index < 0 or substep_index >= num_choices:
        raise ConfigurationError(
            f"Substep index {substep_index} is out of range for step with {num_choices} variants"
        )

    namespace = copy.copy(common_constants)
    namespace.update(step.get('constants', {}))
    for key, values in variables.items():
        namespace[key] = values[substep_index]

    result = copy.deepcopy(step)
    result['parameters'] = expand_step_parameters(step.get('parameters', {}), namespace)
    result['variables'] = {}
    return result


def expand_steps_with_variables(steps, common_constants=None):
    """Expand steps with variables into individual substeps.

    Args:
        steps: List of step configurations
        common_constants: Dictionary of common constants from config (optional)

    Returns a list of expanded steps. Steps without variables are returned
    as-is. Steps with variables are expanded into multiple substeps, each
    with resolved parameters.

    Each expanded step includes:
    - _original_index: original step index
    - _substep_index: index within variable combinations (None if no variables)
    - _expanded_parameters: parameters with variables resolved
    - _expanded_depends_on: depends_on field with variables resolved

    For zipped dependencies (same step with variables), substep B[idx]
    corresponds to substep A[idx] from the previous step.
    """
    common_constants = common_constants or {}
    expanded = []
    for original_idx, step in enumerate(steps):
        variables = step.get('variables', {})
        if not variables:
            expanded_step = copy.deepcopy(step)
            expanded_step['_original_index'] = original_idx
            expanded_step['_substep_index'] = None
            namespace = copy.copy(common_constants)
            namespace.update(step.get('constants', {}))
            expanded_step['_expanded_parameters'] = expand_step_parameters(
                step.get('parameters', {}), namespace
            )
            if 'depends_on' in step:
                expanded_step['_expanded_depends_on'] = expand_step_parameters(
                    step['depends_on'], namespace
                )
            expanded.append(expanded_step)
            continue

        lengths = set()
        for key, value in variables.items():
            if not isinstance(value, list):
                raise ConfigurationError(f"Variable {key} does not define a list")
            lengths.add(len(value))
        if len(lengths) > 1:
            raise ConfigurationError(
                f"Variables have inconsistent lengths: {lengths}. "
                "All variables must have the same number of values."
            )
        num_choices = list(lengths)[0] if lengths else 0

        if not num_choices:
            logger.warning(
                f"Step {original_idx} ({step.get('type')}): "
                "variable value lists are empty, skipping"
            )
            continue

        namespace = copy.copy(common_constants)
        namespace.update(step.get('constants', {}))
        for idx in range(num_choices):
            for key, values in variables.items():
                namespace[key] = values[idx]

            expanded_step = copy.deepcopy(step)
            expanded_step['_original_index'] = original_idx
            expanded_step['_substep_index'] = idx
            expanded_step['_expanded_parameters'] = expand_step_parameters(
                step.get('parameters', {}), namespace
            )
            if 'depends_on' in step:
                expanded_step['_expanded_depends_on'] = expand_step_parameters(
                    step['depends_on'], namespace
                )
            expanded.append(expanded_step)
            logger.debug(
                f"Expanded step {original_idx} ({step.get('type')}) "
                f"substep {idx}: {namespace}"
            )

    return expanded
