import json
import logging
import os
import shutil
import tempfile
import unittest


from opusfilter.opusfilter import OpusFilter
from opusfilter.util import *


class TestYAML(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_simple(self):
        config_string = """
common:
  output_directory: %s
steps:
  - type: write
    parameters:
      output: input.en
      data: Hello world!

  - type: write
    parameters:
      output: input.fi
      data: Hei maailma!

  - type: concatenate
    parameters:
      inputs:
      - input.fi
      - input.en
      output: output.txt
""" % self.tempdir
        config = yaml.load(config_string)
        of = OpusFilter(config)
        of.execute_steps()
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir, 'output.txt')))

    def test_variables(self):
        config_string = """
common:
  output_directory: %s
  constants:
    l1: fi
steps:
  - type: write
    parameters:
      output: input.en
      data: |
         .
         Hello world!
         So long, and thanks for all the fish

  - type: write
    parameters:
      output: input.fi
      data: |
         .
         Hei maailma!
         Terve, ja kiitos kaloista

  - type: write
    parameters:
      output: input.sv
      data: |
         .
         Hej världen!
         Ajöss och tack för fisken

  - type: filter
    parameters:
      inputs:
      - !varstr "input.{l1}"
      - !varstr "input.{l2}"
      outputs:
      - !varstr "filtered.{l1}-{l2}.{l1}"
      - !varstr "filtered.{l1}-{l2}.{l2}"
      filters:
      - LengthFilter:
          unit: word
          min_length: !var minlen
          max_length: !var maxlen
    constants:
      minlen: 1
    variables:
      l2: [en, sv]
      maxlen: [10, 6]
""" % self.tempdir
        config = yaml.load(config_string)
        of = OpusFilter(config)
        of.execute_steps()
        for outfile in ('filtered.fi-en.fi', 'filtered.fi-en.en',
                        'filtered.fi-sv.fi', 'filtered.fi-sv.sv'):
            self.assertTrue(os.path.isfile(os.path.join(self.tempdir, outfile)))
