"""Tests for opusfilter-scores CLI command."""
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, mock_open
import json

from opusfilter.cli.scores import main as scores_main


class TestScoresCLI(unittest.TestCase):
    """Test cases for opusfilter-scores command."""

    def test_scores_help(self):
        """Test that help is displayed."""
        with self.assertRaises(SystemExit) as cm:
            scores_main(['--help'])
        self.assertEqual(cm.exception.code, 0)

    def test_scores_list_subcommand_help(self):
        """Test list subcommand help."""
        with self.assertRaises(SystemExit) as cm:
            scores_main(['list', '--help'])
        self.assertEqual(cm.exception.code, 0)

    @patch('opusfilter.cli.scores.file_open', mock_open(read_data='{"score": 1.0}\n{"score": 2.0}\n'))
    def test_scores_list_command(self):
        """Test list subcommand."""
        result = scores_main(['list', 'test.jsonl'])
        self.assertEqual(result, 0)

    @patch('opusfilter.cli.scores.file_open', mock_open(read_data='{"score": 1.0}\n{"score": 2.0}\n'))
    @patch('opusfilter.cli.scores.plt.show')
    def test_scores_describe_command(self, mock_show):
        """Test describe subcommand."""
        # Don't show plots for describe command
        result = scores_main(['describe', 'test.jsonl'])
        self.assertEqual(result, 0)
        self.assertFalse(mock_show.called)

    def test_invalid_command(self):
        """Test invalid command."""
        with self.assertRaises(SystemExit) as cm:
            scores_main(['invalid-command'])
        self.assertEqual(cm.exception.code, 1)

    def test_installed_command(self):
        """Test that the installed command works."""
        result = subprocess.run(
            [sys.executable, '-m', 'opusfilter.cli.scores', '--help'],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn('Plot and diagnose filter scores', result.stdout)


if __name__ == '__main__':
    unittest.main()