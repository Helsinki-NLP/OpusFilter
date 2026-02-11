import unittest
import tempfile
import os
from unittest import mock
from opusfilter import ConfigurationError


class TestSlurmUtils(unittest.TestCase):
    """Test SLURM utility functions."""
    
    def test_build_dependency_graph(self):
        """Test dependency graph building."""
        from opusfilter.slurm_utils import build_dependency_graph
        
        steps = [
            {
                'type': 'opus_read',
                'parameters': {
                    'outputs': ['test.en']
                }
            },
            {
                'type': 'filter',
                'parameters': {
                    'inputs': ['test.en'],
                    'outputs': ['filtered.en']
                }
            }
        ]
        
        graph = build_dependency_graph(steps)
        
        # Check that filter depends on opus_read
        filter_deps = graph['1_filter']['deps']
        self.assertIn('0_opus_read', filter_deps)
        
        # Check that opus_read has no dependencies
        opus_read_deps = graph['0_opus_read'].get('deps', [])
        
        # Check that opus_read has no dependencies
        opus_read_deps = graph['0_opus_read']['deps']
        self.assertEqual(len(opus_read_deps), 0)
    
    def test_get_ready_steps(self):
        """Test getting ready steps."""
        from opusfilter.slurm_utils import get_ready_steps
        
        graph = {
            '0_opus_read': {'deps': [], 'completed': False},
            '1_filter': {
                'deps': ['0_opus_read'],
                'completed': False
            },
            '2_train': {
                'deps': [],
                'completed': False
            }
        }
        
        # Test with no completed jobs - only opus_read and train have no deps
        ready = get_ready_steps(graph, [])
        self.assertEqual(len(ready), 2)
        self.assertIn('0_opus_read', ready)
        self.assertIn('2_train', ready)
        
        # Test with completed opus_read - create fresh graph
        graph2 = {
            '0_opus_read': {'deps': [], 'completed': True},
            '1_filter': {
                'deps': ['0_opus_read'],
                'completed': False
            },
            '2_train': {
                'deps': [],
                'completed': False
            }
        }
        ready = get_ready_steps(graph2, ['0_opus_read'])
        # 0_opus_read should not be returned because it's marked as completed
        # 1_filter needs opus_read which is completed, so it's ready
        # 2_train has no deps, so it's ready
        self.assertEqual(len(ready), 2)
        self.assertIn('1_filter', ready)
        self.assertIn('2_train', ready)


if __name__ == '__main__':
    unittest.main()