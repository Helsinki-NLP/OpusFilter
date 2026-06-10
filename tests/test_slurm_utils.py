import unittest
import tempfile
import os
from unittest import mock

from opusfilter.cli.slurm import main
from opusfilter.slurm import SlurmOpusFilter
from opusfilter.opusfilter import OpusFilter


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
        filter_deps = graph['2_filter']['deps']
        self.assertIn('1_opus_read', filter_deps)
        
        # Check that opus_read has no dependencies
        opus_read_deps = graph['1_opus_read'].get('deps', [])
        self.assertEqual(len(opus_read_deps), 0)
    
    def test_get_ready_steps(self):
        """Test getting ready steps."""
        from opusfilter.slurm_utils import get_ready_steps
        
        graph = {
            '1_opus_read': {'deps': [], 'completed': False},
            '2_filter': {
                'deps': ['1_opus_read'],
                'completed': False
            },
            '3_train': {
                'deps': [],
                'completed': False
            }
        }
        
        # Test with no completed jobs - only opus_read and train have no deps
        ready = get_ready_steps(graph, [])
        self.assertEqual(len(ready), 2)
        self.assertIn('1_opus_read', ready)
        self.assertIn('3_train', ready)
        
        # Test with completed opus_read - create fresh graph
        graph2 = {
            '1_opus_read': {'deps': [], 'completed': True},
            '2_filter': {
                'deps': ['1_opus_read'],
                'completed': False
            },
            '3_train': {
                'deps': [],
                'completed': False
            }
        }
        ready = get_ready_steps(graph2, ['1_opus_read'])
        self.assertEqual(len(ready), 2)
        self.assertIn('2_filter', ready)
        self.assertIn('3_train', ready)
        
        # Test with completed opus_read and filter ready
        graph3 = {
            '1_opus_read': {'deps': [], 'completed': True},
            '2_filter': {
                'deps': ['1_opus_read'],
                'completed': True
            },
            '3_train': {
                'deps': [],
                'completed': False
            }
        }
        ready = get_ready_steps(graph3, ['1_opus_read', '2_filter'])
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0], '3_train')

    def test_expand_steps_with_variables_depends_on(self):
        """Test that depends_on field is expanded correctly."""
        from opusfilter.util import expand_steps_with_variables

        steps = [
            {
                'type': 'train_ngram',
                'parameters': {
                    'model': 'lang.arpa.gz'
                },
                'variables': {
                    'lang': ['en', 'de', 'fr']
                }
            },
            {
                'type': 'filter',
                'parameters': {
                    'inputs': ['data.en.gz']
                },
                'depends_on': ['en.arpa.gz', 'de.arpa.gz', 'fr.arpa.gz']
            }
        ]

        expanded = expand_steps_with_variables(steps)

        # Should have 3 expanded train steps + 1 filter step
        self.assertEqual(len(expanded), 4)

        # Check filter step has expanded depends_on
        filter_step = expanded[3]
        self.assertIn('_expanded_depends_on', filter_step)
        self.assertEqual(filter_step['_expanded_depends_on'], ['en.arpa.gz', 'de.arpa.gz', 'fr.arpa.gz'])

    def test_build_dependency_graph_depends_on(self):
        """Test explicit depends_on dependencies."""
        from opusfilter.slurm_utils import build_dependency_graph

        steps = [
            {
                'type': 'train_ngram',
                'parameters': {
                    'model': 'en.arpa.gz'
                }
            },
            {
                'type': 'train_ngram',
                'parameters': {
                    'model': 'de.arpa.gz'
                }
            },
            {
                'type': 'filter',
                'parameters': {
                    'inputs': ['data.en.gz'],
                    'outputs': ['filtered.en.gz']
                },
                'depends_on': ['en.arpa.gz']
            }
        ]

        graph = build_dependency_graph(steps)

        # Filter should depend on train_ngram via depends_on
        filter_deps = graph['3_filter']['deps']
        self.assertIn('1_train_ngram', filter_deps)
        # And also from input/output matching
        self.assertEqual(len(filter_deps), 1)

    def test_build_dependency_graph_depends_on_variables(self):
        """Test depends_on with variable expansion."""
        from opusfilter.util import expand_steps_with_variables
        from opusfilter.slurm_utils import build_dependency_graph

        steps = [
            {
                'type': 'train_ngram',
                'parameters': {
                    'model': '!varstr "{lang}.arpa.gz"'
                },
                'variables': {
                    'lang': ['en', 'de']
                }
            },
            {
                'type': 'filter',
                'parameters': {
                    'inputs': ['data.gz']
                },
                'depends_on': ['!varstr "{lang}.arpa.gz"'],
                'variables': {
                    'lang': ['en', 'de']
                }
            }
        ]

        expanded = expand_steps_with_variables(steps)
        graph = build_dependency_graph(expanded)

        # filter_0 should depend on train_ngram_0 (both en)
        filter_0_deps = graph['2_filter_1']['deps']
        self.assertIn('1_train_ngram_1', filter_0_deps)

        # filter_1 should depend on train_ngram_1 (both de)
        filter_1_deps = graph['2_filter_2']['deps']
        self.assertIn('1_train_ngram_2', filter_1_deps)


if __name__ == '__main__':
    unittest.main()