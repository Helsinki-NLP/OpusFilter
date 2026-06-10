"""Tests for opusfilter-duplicates CLI command."""
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, mock_open
import json
import io

from opusfilter.cli.duplicates import main as duplicates_main


class TestDuplicatesCLI(unittest.TestCase):
    """Test cases for opusfilter-duplicates command."""

    def test_duplicates_help(self):
        """Test that help is displayed."""
        with self.assertRaises(SystemExit) as cm:
            duplicates_main(['--help'])
        self.assertEqual(cm.exception.code, 0)

    @patch('opusfilter.util.file_open')
    def test_duplicates_with_duplicates(self, mock_open):
        """Test duplicate detection with actual duplicates."""
        # Mock files with duplicate segments
        mock_file1 = io.StringIO('segment1\nsegment2\n')
        mock_file2 = io.StringIO('segment2\nsegment3\n')
        mock_file3 = io.StringIO('segment1\n')
        mock_open.side_effect = [mock_file1, mock_file2, mock_file3]
        result = duplicates_main(['file1.txt', 'file2.txt', 'file3.txt'])
        self.assertEqual(result, 0)
        # Check that three files were opened
        self.assertEqual(mock_open.call_count, 3)

    @patch('opusfilter.util.file_open')
    def test_overlap_statistics(self, mock_open):
        # Setup mock files for first set
        mock_file1 = io.StringIO('segment1\nsegment2\n')
        mock_file2 = io.StringIO('segment2\nsegment3\n')
        # Setup mock files for second set (overlap)
        mock_file3 = io.StringIO('segment2\nsegment3\n')
        mock_file4 = io.StringIO('segment5\nsegment6\n')
        mock_open.side_effect = [mock_file1, mock_file2, mock_file3, mock_file4]
        result = duplicates_main(['file1.txt', 'file2.txt', '--overlap', 'file3.txt', 'file4.txt'])
        self.assertEqual(result, 0)

    def test_installed_command(self):
        """Test that installed command works."""
        result = subprocess.run(
            [sys.executable, '-m', 'opusfilter.cli.duplicates', '--help'],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn('Find duplicates', result.stdout)


if __name__ == '__main__':
    unittest.main()
