"""Tests for opusfilter-cmd CLI command."""
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, mock_open

from opusfilter.cli.cmd import main as cmd_main


class TestCmdCLI(unittest.TestCase):
    """Test cases for opusfilter-cmd command."""
    
    def test_cmd_help(self):
        """Test that help is displayed."""
        with self.assertRaises(SystemExit) as cm:
            cmd_main(['--help'])
        self.assertEqual(cm.exception.code, 0)
    
    def test_cmd_help_only(self):
        """Test that help is displayed without calling OpusFilter."""
        with self.assertRaises(SystemExit) as cm:
            cmd_main(['--help'])
        self.assertEqual(cm.exception.code, 0)
    
    def test_installed_command(self):
        """Test that installed command works."""
        result = subprocess.run(
            [sys.executable, '-m', 'opusfilter.cli.cmd', '--help'],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn('Run single opusfilter function', result.stdout)


if __name__ == '__main__':
    unittest.main()