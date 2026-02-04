import unittest
import tempfile
import os
from unittest import mock

from opusfilter.util import yaml
from opusfilter.cli.slurm import main
from opusfilter.slurm import SlurmOpusFilter
from opusfilter.opusfilter import OpusFilter


class TestSlurmIntegration(unittest.TestCase):
    """Test SLURM integration functionality."""
    
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.config = {
            'common': {
                'output_directory': 'test_output',
                'slurm': {
                    'account': 'testaccount',
                    'partition': 'cpu',
                    'mail_user': 'test@example.com',
                    'default': {
                        'time': '01:00:00',
                        'mem': '2G',
                        'cpus-per-task': 1
                    },
                    'resources': {
                        'opus_read': {
                            'time': '00:30:00',
                            'mem': '1G'
                        },
                        'filter': {
                            'time': '01:00:00',
                            'mem': '2G',
                            'array_size': 2
                        }
                    }
                }
            },
            'steps': [
                {
                    'type': 'opus_read',
                    'parameters': {
                        'corpus_name': 'test',
                        'source_language': 'en',
                        'target_language': 'fi',
                        'src_output': 'test.en',
                        'tgt_output': 'test.fi'
                    }
                },
                {
                    'type': 'filter',
                    'parameters': {
                        'inputs': ['test.en', 'test.fi'],
                        'outputs': ['filtered.en', 'filtered.fi'],
                        'filters': [
                            {'LengthFilter': {'min_length': 1}}
                        ]
                    }
                }
            ]
        }
        
        # Write config file
        self.config_file = os.path.join(self.tempdir, 'test_config.yaml')
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.tempdir)
    
    @mock.patch('opusfilter.slurm_utils.submit_job')
    @mock.patch('opusfilter.slurm_utils.get_job_status')
    def test_dry_run(self, mock_submit, mock_status):
        """Test dry run mode."""
        mock_submit.return_value = '12345'
        mock_status.return_value = 'PENDING'
        
        with mock.patch('sys.argv', ['opusfilter-slurm', self.config_file, '--dry-run']):
            main()
        
        # Check that job was submitted with correct parameters
        mock_submit.assert_called()
        call_args = mock_submit.call_args[0]
        script_path = call_args[0][0]
        
        # Verify script contains correct configuration
        with open(script_path.replace('.sbatch', '.template')) as f:
            content = f.read()
            self.assertIn('--time=00:30:00', content)
            self.assertIn('--mem=1G', content)
            self.assertIn('job_name=0_opus_read', content)
    
    def test_resource_inheritance(self):
        """Test that step-specific resources override defaults."""
        slurm_filter = SlurmOpusFilter(self.config)
        
        # Check opus_read step uses specific resources
        opus_read_resources = slurm_filter._get_step_resources('opus_read')
        self.assertEqual(opus_read_resources['time'], '00:30:00')
        self.assertEqual(opus_read_resources['mem'], '1G')
        
        # Check filter step uses specific resources
        filter_resources = slurm_filter._get_step_resources('filter')
        self.assertEqual(filter_resources['time'], '01:00:00')
        self.assertEqual(filter_resources['mem'], '2G')
        self.assertEqual(filter_resources['array_size'], 2)
    
    def test_dependency_detection(self):
        """Test automatic dependency detection."""
        from opusfilter.slurm_utils import build_dependency_graph
        
        graph = build_dependency_graph(self.config['steps'])
        
        # Check that filter depends on opus_read
        filter_deps = graph['1_filter']['deps']
        self.assertIn('0_opus_read', filter_deps)
        
        # Check that opus_read has no dependencies
        opus_read_deps = graph['0_opus_read']['deps']
        self.assertEqual(len(opus_read_deps), 0)


if __name__ == '__main__':
    unittest.main()