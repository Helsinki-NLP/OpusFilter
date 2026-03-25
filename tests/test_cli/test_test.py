"""Tests for opusfilter-test CLI command."""
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock, mock_open
import json

from opusfilter.cli.test import main as cli_main


class TestOpusfilterTestCLI(unittest.TestCase):
    """Test cases for opusfilter-test command."""

    def test_help(self):
        """Test that help is displayed."""
        with self.assertRaises(SystemExit) as cm:
            cli_main(['--help'])
        self.assertEqual(cm.exception.code, 0)

    def test_with_add(self):
        """Test with a real filter configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as data_file:
            data_file.write('Test sentence 1')
            data_file.flush()
            result = cli_main([data_file.name, '--add', 'LengthFilter', '{"min_length": 5}'])
            self.assertEqual(result, 0)

    def test_with_yaml(self):
        """Test test with YAML configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as config_file, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as data_file:
            config_file.write('[LengthFilter: {}]')
            config_file.flush()
            data_file.write('Test sentence 1')
            data_file.flush()
            result = cli_main(['--yaml', config_file.name, data_file.name])
            self.assertEqual(result, 0)

    def test_installed_command(self):
        """Test that installed command works."""
        result = subprocess.run(
            [sys.executable, '-m', 'opusfilter.cli.test', '--help'],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn('Test filters on', result.stdout)


if __name__ == '__main__':
    unittest.main()
