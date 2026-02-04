"""Tests for opusfilter-diagram CLI command."""
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, mock_open, MagicMock
import os

from opusfilter.cli.diagram import main as diagram_main


class TestDiagramCLI(unittest.TestCase):
    """Test cases for opusfilter-diagram command."""

    def test_diagram_help_only(self):
        """Test that help is displayed."""
        with self.assertRaises(SystemExit) as cm:
            diagram_main(['--help'])
        self.assertEqual(cm.exception.code, 0)

    @patch('opusfilter.cli.diagram.open')
    @patch('opusfilter.cli.diagram.yaml.load')
    def test_diagram_with_real_config(self, mock_yaml, mock_open):
        """Test diagram generation with a real config structure."""
        # Create a mock config with proper structure
        mock_config = {
            'steps': [
                {
                    'type': 'read_from_opus',
                    'parameters': {
                        'src_lang': 'en',
                        'tgt_lang': 'fi',
                        'corpus_name': 'OpenSubtitles'
                    }
                },
                {
                    'type': 'filter',
                    'parameters': {
                        'inputs': ['input.gz'],
                        'outputs': ['filtered.gz'],
                        'Filter': 'WordRatioFilter',
                        'parameters': {'threshold': 0.1}
                    }
                }
            ]
        }
        mock_yaml.return_value = mock_config
        read_handle = mock_open().return_value.__enter__.return_value
        write_handle = MagicMock()
        mock_open.side_effect = [read_handle, write_handle]

        result = diagram_main(['config.yaml', 'output.dot'])
        self.assertEqual(result, 0)

        mock_open.assert_any_call('config.yaml', 'r', encoding='utf-8')
        mock_yaml.assert_called_once_with(read_handle)
        mock_open.assert_any_call('output.dot', 'w', encoding='utf-8')
        dot_fobj = write_handle.__enter__()
        dot_fobj.write.assert_called_once()
        args, _ = dot_fobj.write.call_args
        self.assertIn('digraph', args[0])

    def test_installed_command(self):
        """Test that installed command works."""
        result = subprocess.run(
            [sys.executable, '-m', 'opusfilter.cli.diagram', '--help'],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn('Draw a diagram', result.stdout)


if __name__ == '__main__':
    unittest.main()
