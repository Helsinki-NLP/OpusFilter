import unittest
import tempfile
import os
from unittest import mock
from unittest.mock import Mock

from opusfilter import ConfigurationError


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
            f.write('Hei maailma\nTämä on suomi\n')
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.tempdir)
    
    @mock.patch('opusfilter.slurm_utils.submit_job')
    @mock.patch('opusfilter.slurm_utils.get_job_status')
    @mock.patch('time.sleep')
    def test_simple_workflow(self, mock_submit, mock_status, mock_sleep):
        """Test a simple workflow execution."""
        from opusfilter.cli.slurm import main
        from opusfilter.opusfilter import OpusFilter
        
        # Create config
        config = {
            'common': {
                'output_directory': self.tempdir,
                'slurm': {
                    'account': 'test',
                    'partition': 'cpu',
                    'default': {
                        'time': '00:05:00',
                        'mem': '1G'
                    }
                }
            },
            'steps': [
                {
                    'type': 'filter',
                    'parameters': {
                        'inputs': [os.path.basename(self.en_file), os.path.basename(self.fi_file)],
                        'outputs': ['filtered.en', 'filtered.fi']
                    }
                }
            ]
        }
        
        config['_config_file'] = os.path.join(self.tempdir, 'test_config.yaml')
        
        # Mock job completion
        mock_submit.return_value = 'job1'
        mock_status.side_effect = ['RUNNING', 'COMPLETED']
        mock_sleep.side_effect = [None, None]  # Don't actually sleep
        
        # Run with dry run first
        with mock.patch('sys.argv', ['opusfilter-slurm', 'test_config.yaml', '--dry-run']):
            try:
                main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)
        
        # Check that submit was called correctly
                mock_submit.assert_called_once()
        mock_submit.assert_called_once()
        call_args = mock_submit.call_args[0]
        script_path = call_args[0][0]
        
        # Verify the script exists and has correct content
        with open(script_path, 'r') as f:
            content = f.read()
            self.assertIn('--time=00:05:00', content)
            self.assertIn('--job_name=0_filter', content)
        
        # Now test actual execution
        mock_submit.reset_mock()
        mock_status.reset_mock()
        mock_status.side_effect = None  # No sleeping
        
        with mock.patch('sys.argv', ['opusfilter-slurm', 'test_config.yaml']):
            with mock.patch('opusfilter.slurm.SlurmOpusFilter') as MockSlurm:
                mock_instance = MockSlurm()
                mock_instance.run.return_value = None
                main()
                
                # Verify run was called
                mock_instance.run.assert_called_once_with(
                    overwrite=False, resume=False, monitor=False
                )


if __name__ == '__main__':
    unittest.main()