"""Tests for opusfilter-autogen CLI command."""
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, mock_open

from opusfilter.cli.autogen import main as autogen_main


class TestAutogenCLI(unittest.TestCase):
    """Test cases for opusfilter-autogen command."""

    def test_autogen_help(self):
        """Test that help is displayed."""
        with self.assertRaises(SystemExit) as cm:
            autogen_main(['--help'])
        self.assertEqual(cm.exception.code, 0)

    def test_autogen_requires_files(self):
        """Test that files argument is required."""
        with self.assertRaises(SystemExit) as cm:
            autogen_main([])
        self.assertEqual(cm.exception.code, 2)

    def test_installed_command(self):
        """Test that installed command works."""
        result = subprocess.run(
            [sys.executable, '-m', 'opusfilter.cli.autogen', '--help'],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn('Generate initial configuration', result.stdout)


if __name__ == '__main__':
    unittest.main()
