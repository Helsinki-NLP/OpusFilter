import json
import logging
import os
import shutil
import tempfile
import unittest

from opusfilter import ConfigurationError
from opusfilter.util import *


class TestCheckArgsCompability(unittest.TestCase):

    def test_value(self):
        newvalue = check_args_compability(1, required_types=[int])
        self.assertEqual(newvalue[0], 1)

    def test_list(self):
        newvalue = check_args_compability([1, 1, 2], required_types=[int])
        self.assertSequenceEqual(newvalue, [1, 1, 2])

    def test_single_values(self):
        args = {'min_length': 2, 'max_length': 10, 'threshold': 1}
        names, values = zip(*args.items())
        newvalues = check_args_compability(*values, required_types=[int, int, int], names=names)
        self.assertSequenceEqual([v[0] for v in newvalues], values)

    def test_list_values(self):
        args = {'min_length': [2, 3], 'max_length': [10, 15], 'threshold': [1, 1]}
        names, values = zip(*args.items())
        newvalues = check_args_compability(*values, required_types=[int, int, int], names=names)
        self.assertSequenceEqual(newvalues, values)

    def test_list_values_any_type(self):
        args = {'min_length': [2, 33.5], 'max_length': [1.1, 15], 'threshold': ['a', 3]}
        names, values = zip(*args.items())
        newvalues = check_args_compability(*values, names=names)
        self.assertSequenceEqual(newvalues, values)

    def test_list_and_single_values(self):
        args = {'min_length': 1, 'max_length': 10, 'threshold': [1, 1]}
        names, values = zip(*args.items())
        newvalues = check_args_compability(*values, required_types=[int, int, int], names=names)
        self.assertEqual(len(newvalues), len(values))

    def test_choices_single(self):
        args = {'pass': 'yes', 'threshold': 1}
        names, values = zip(*args.items())
        newvalues = check_args_compability(
            *values, required_types=[str, int], choices=[('yes', 'no'), None], names=names)
        self.assertSequenceEqual([v[0] for v in newvalues], values)

    def test_choices_list(self):
        args = {'pass': ['yes', 'no'], 'threshold': [1, 1]}
        names, values = zip(*args.items())
        newvalues = check_args_compability(
            *values, required_types=[str, int], choices=[('yes', 'no'), None], names=names)
        self.assertSequenceEqual(newvalues, values)

    def test_wrong_type(self):
        args = {'min_length': 2, 'max_length': 10, 'threshold': 0.12}
        names, values = zip(*args.items())
        with self.assertRaises(ConfigurationError):
            check_args_compability(*values, required_types=[int, int, int], names=names)

    def test_wrong_length(self):
        args = {'min_length': [2, 3, 2], 'max_length': [10, 15, 10], 'threshold': [1, 1]}
        names, values = zip(*args.items())
        with self.assertRaises(ConfigurationError):
            check_args_compability(*values, required_types=[int, int, int], names=names)


class TestYAMLDumps(unittest.TestCase):

    def test_yaml_dumps_list(self):
        obj = ['a', 'b', 'c', 1, 2, 3]
        string = yaml_dumps(obj)
        obj2 = yaml.load(string)
        self.assertSequenceEqual(obj, obj2)


class TestGetInputsOutputs(unittest.TestCase):

    def test_get_inputs_basic(self):
        step = {
            'parameters': {
                'inputs': ['file1.txt', 'file2.txt']
            }
        }
        inputs = get_inputs(step)
        self.assertEqual(inputs, ['file1.txt', 'file2.txt'])

    def test_get_inputs_single(self):
        step = {
            'parameters': {
                'input': 'single.txt'
            }
        }
        inputs = get_inputs(step)
        self.assertEqual(inputs, ['single.txt'])

    def test_get_inputs_nested(self):
        step = {
            'parameters': {
                'inputs': [['file1.txt', 'file2.txt'], ['file3.txt']]
            }
        }
        inputs = get_inputs(step)
        self.assertEqual(inputs, ['file1.txt', 'file2.txt', 'file3.txt'])

    def test_get_inputs_depends_on(self):
        step = {
            'parameters': {
                'inputs': ['data.txt']
            },
            'depends_on': ['model1.arpa.gz', 'model2.arpa.gz']
        }
        inputs = get_inputs(step)
        self.assertEqual(inputs, ['data.txt', 'model1.arpa.gz', 'model2.arpa.gz'])

    def test_get_inputs_depends_on_string(self):
        step = {
            'parameters': {
                'inputs': ['data.txt']
            },
            'depends_on': 'single.arpa.gz'
        }
        inputs = get_inputs(step)
        self.assertEqual(inputs, ['data.txt', 'single.arpa.gz'])

    def test_get_outputs_basic(self):
        step = {
            'parameters': {
                'outputs': ['out1.txt', 'out2.txt']
            }
        }
        outputs = get_outputs(step)
        self.assertEqual(outputs, ['out1.txt', 'out2.txt'])

    def test_get_outputs_single(self):
        step = {
            'parameters': {
                'output': 'single.txt'
            }
        }
        outputs = get_outputs(step)
        self.assertEqual(outputs, ['single.txt'])

    def test_get_outputs_training(self):
        step = {
            'type': 'train_ngram',
            'parameters': {
                'data': 'data.txt',
                'model': 'model.arpa.gz'
            }
        }
        outputs = get_outputs(step)
        self.assertEqual(outputs, ['model.arpa.gz'])
