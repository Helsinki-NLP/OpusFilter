import unittest
import tempfile
import os
from unittest import mock
from unittest.mock import Mock

from opusfilter import ConfigurationError
from opusfilter.util import yaml


class TestSlurmEndToEnd(unittest.TestCase):
    """Test end-to-end SLURM workflow."""

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()

        # Create test input files
        self.en_file = os.path.join(self.tempdir, 'test.en')
        self.fi_file = os.path.join(self.tempdir, 'test.fi')

        with open(self.en_file, 'w') as f:
            f.write('Hello world\nThis is English\n')
        with open(self.fi_file, 'w') as f:
            f.write('Hei maailma\nTämä on suomea\n')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tempdir)

    def test_simple_workflow(self):
        """Test a simple workflow execution."""
        # Just a minimal test to ensure imports work
        from opusfilter.cli.slurm import main
        from opusfilter.opusfilter import OpusFilter

        # Create minimal config
        config = {
            'common': {
                'output_directory': self.tempdir,
                'slurm': {
                    'account': 'test',
                    'partition': 'cpu'
                }
            },
            'steps': []
        }

        # Write config file
        config_file = os.path.join(self.tempdir, 'test_config.yaml')
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        # Just test that the CLI can parse args
        with mock.patch('sys.argv', ['opusfilter-slurm', config_file, '--dry-run']):
            with mock.patch('opusfilter.cli.slurm.SlurmOpusFilter') as MockSlurm:
                mock_instance = MockSlurm()
                mock_instance.run.return_value = None

                try:
                    main()
                except SystemExit as e:
                    self.assertEqual(e.code, 0)

                # Verify run was called
                mock_instance.run.assert_called_once_with(
                    overwrite=False, resume=False, monitor=False
                )


if __name__ == '__main__':
    unittest.main()
