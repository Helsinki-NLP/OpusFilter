"""Tests for opusfilter CLI command."""
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, mock_open

from opusfilter.cli.main import main as opusfilter_main


class TestOpusFilterCLI(unittest.TestCase):
    """Test cases for opusfilter command."""

    def test_main_help(self):
        """Test that help is displayed."""
        with self.assertRaises(SystemExit) as cm:
            opusfilter_main(['--help'])
        self.assertEqual(cm.exception.code, 0)

    @patch('opusfilter.cli.main.open', mock_open(read_data='steps: []'))
    @patch('opusfilter.cli.main.OpusFilter')
    def test_main_with_config(self, mock_opusfilter):
        """Test running with a config file."""
        result = opusfilter_main(['test.yaml'])
        self.assertEqual(result, 0)
        mock_opusfilter.assert_called_once()

    def test_installed_command(self):
        """Test that the installed command works."""
        result = subprocess.run(
            [sys.executable, '-m', 'opusfilter.cli.main', '--help'],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn('Filter OPUS bitexts', result.stdout)


if __name__ == '__main__':
    unittest.main()