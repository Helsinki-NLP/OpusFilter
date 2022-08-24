import copy
import json
import logging
import os
import requests
import shutil
import tempfile
import unittest
from argparse import Namespace
from unittest import mock

from opustools import OpusGet

from opusfilter import ConfigurationError
from opusfilter.opusfilter import OpusFilter, ParallelWrapper
from opusfilter.util import Var, VarStr, count_lines, file_open

try:
    import varikn
except ImportError:
    logging.warning("Could not load varikn, language model filtering tests not supported")


@unittest.skipIf('varikn' not in globals() or os.environ.get('EFLOMAL_PATH') is None, 'varikn or eflomal not found')
class TestOpusFilter(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tempdir = tempfile.mkdtemp()
        self.configuration = {
            'common': {'output_directory': self.tempdir},
            'steps':
            [{'type': 'opus_read',
              'parameters': {'corpus_name': 'RF',
                             'source_language': 'en',
                             'target_language': 'sv',
                             'release': 'latest',
                             'preprocessing': 'xml',
                             'src_output': 'RF1_sents.en',
                             'tgt_output': 'RF1_sents.sv'}},
             {'type': 'filter',
              'parameters': {
                  'inputs': ['RF1_sents.en', 'RF1_sents.sv'],
                  'outputs': ['RF1_filtered.en', 'RF1_filtered.sv'],
                  'filters': [{'LanguageIDFilter':
                               {'languages': ['en', 'sv'],
                                'thresholds': [0, 0]}},
                              {'TerminalPunctuationFilter':
                               {'threshold': -2}},
                              {'NonZeroNumeralsFilter': {'threshold': 0.5}},
                              {'CharacterScoreFilter':
                               {'scripts': ['Latin', 'Latin'],
                                'thresholds': [1, 1]}}]}},
             {'type': 'train_ngram',
              'parameters': {'data': 'RF1_filtered.en',
                             'parameters': {'norder': 20, 'dscale': 0.001},
                             'model': 'RF1_en.arpa'}},
             {'type': 'train_ngram',
              'parameters': {'data': 'RF1_filtered.sv',
                             'parameters': {'norder': 20, 'dscale': 0.001},
                             'model': 'RF1_sv.arpa'}},
             {'type': 'train_alignment',
              'parameters': {'src_data': 'RF1_filtered.en',
                             'tgt_data': 'RF1_filtered.sv',
                             'parameters': {'model': 3},
                             'output': 'RF1_align.priors'}},
             {'type': 'score',
              'parameters': {
                  'inputs': ['RF1_sents.en', 'RF1_sents.sv'],
                  'output': 'RF1_scores.en-sv.jsonl',
                  'filters': [{'LanguageIDFilter':
                               {'languages': ['en', 'sv'],
                                'thresholds': [0, 0]}},
                              {'TerminalPunctuationFilter':
                               {'threshold': -2}},
                              {'NonZeroNumeralsFilter': {'threshold': 0.5}},
                              {'CharacterScoreFilter':
                               {'scripts': ['Latin', 'Latin'],
                                'sthreshold': [1, 1]}},
                              {'WordAlignFilter': {'priors': 'RF1_align.priors',
                                                   'model': 3,
                                                   'src_threshold': 0,
                                                   'tgt_threshold': 0}},
                              {'CrossEntropyFilter':
                               {'lm_params': [{'filename': 'RF1_en.arpa'},
                                              {'filename': 'RF1_sv.arpa'}],
                                'thresholds': [50.0, 50.0],
                                'diff_threshold': 10.0}}]}},
             {'type': 'train_classifier',
              'parameters': {
                  'training_scores': 'RF1_scores.en-sv.jsonl',
                  'model': 'classifier.bin',
                  'criterion': 'CE',
                  'features': {
                      'LanguageIDFilter': {'clean-direction': 'high'},
                      'TerminalPunctuationFilter': {'clean-direction': 'high'},
                      'NonZeroNumeralsFilter': {'clean-direction': 'high'},
                      'CharacterScoreFilter': {'clean-direction': 'high'},
                      'WordAlignFilter': {'clean-direction': 'low'},
                      'CrossEntropyFilter': {'clean-direction': 'low'},
                  }}},
             {'type': 'classify',
              'parameters': {
                  'scores': 'RF1_scores.en-sv.jsonl',
                  'model': 'classifier.bin',
                  'output_probabilities': 'RF1_probs.en-sv.txt'
              }}
             ]}

        OpusGet(directory='RF', source='en', target='sv', release='latest',
            preprocess='xml', suppress_prompts=True, download_dir=self.tempdir
            ).get_files()
        self.opus_filter = OpusFilter(self.configuration)
        self.opus_filter.execute_steps()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.tempdir)

    def test_clean_data(self):
        with open(os.path.join(self.tempdir, 'RF1_filtered.en')) as clean:
            self.assertEqual(
                    clean.readline(),
                    'Your Majesties , Your Royal Highnesses , Mr Speaker , '
                    'Members of the Swedish Parliament .\n'
                    )
        with open(os.path.join(self.tempdir, 'RF1_filtered.sv')) as clean:
            self.assertEqual(
                    clean.readline(),
                    'Eders Majestäter , Eders Kungliga Högheter , herr '
                    'talman , ledamöter av Sveriges riksdag !\n'

                    )

    def test_train_models(self):
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir, 'RF1_align.priors')))
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir, 'RF1_en.arpa')))
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir, 'RF1_en.arpa')))
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir, 'classifier.bin')))

    def test_score_data(self):
        with open(os.path.join(self.tempdir, 'RF1_scores.en-sv.jsonl')) as scores_file:
            score = json.loads(scores_file.readline())
            self.assertEqual(score['LanguageIDFilter'], [1.0, 0.98])
            self.assertEqual(score['LanguageIDFilter'], [1.0, 0.98])
            self.assertEqual(score['CharacterScoreFilter'], [1.0, 1.0])
            self.assertAlmostEqual(
                score['CrossEntropyFilter'][0], 15.214258903317491)
            self.assertAlmostEqual(
                score['CrossEntropyFilter'][1], 7.569084909162213)
            self.assertEqual(score['TerminalPunctuationFilter'], -0.0)
            self.assertEqual(score['NonZeroNumeralsFilter'], [0.0])
            self.assertEqual(type(score['WordAlignFilter']), list)

    def test_classifier_probs(self):
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir, 'RF1_probs.en-sv.txt')))
        with open(os.path.join(self.tempdir, 'RF1_probs.en-sv.txt')) as probs_file:
            probs = [float(p.strip()) for p in probs_file.readlines()]
            self.assertTrue(all(0 <= p <= 1 for p in probs))

    def test_initial_files(self):
        with open(os.path.join(self.tempdir, 'RF1_sents.en')) as sents_file_en:
            with open(os.path.join(self.tempdir, 'RF1_sents.sv')) as sents_file_sv:
                sents_en = sents_file_en.readlines()
                sents_sv = sents_file_sv.readlines()
                self.assertEqual(len(sents_en), 180)
                self.assertEqual(len(sents_sv), 180)
                self.assertEqual(
                        sents_en[0],
                        ('Statement of Government Policy by the Prime '
                        'Minister , Mr Ingvar Carlsson , at the Opening '
                        'of the Swedish Parliament on Tuesday , 4 October '
                        ', 1988 .\n')
                        )
                self.assertEqual(sents_sv[0], 'REGERINGSFÖRKLARING .\n')

    @mock.patch('opustools.opus_get.input', create=True)
    def test_write_to_current_dir_if_common_not_specified(self, mocked_input):
        mocked_input.side_effect = ['y']
        step = self.configuration['steps'][0]
        test_config = {'steps': [step]}
        test_filter = OpusFilter(test_config)
        test_filter.execute_steps()
        self.assertTrue(os.path.isfile('RF1_sents.en'))
        self.assertTrue(os.path.isfile('RF1_sents.sv'))
        os.remove('RF1_sents.en')
        os.remove('RF1_sents.sv')
        os.remove('RF_latest_xml_en.zip')
        os.remove('RF_latest_xml_sv.zip')
        os.remove('RF_latest_xml_en-sv.xml.gz')

    @mock.patch('opustools.opus_get.input', create=True)
    def test_write_to_current_dir_if_output_dir_not_specified(self, mocked_input):
        mocked_input.side_effect = ['y']
        common = {'test': 'test'}
        step = self.configuration['steps'][0]
        test_config = {'common': common, 'steps': [step]}
        test_filter = OpusFilter(test_config)
        test_filter.execute_steps()
        self.assertTrue(os.path.isfile('RF1_sents.en'))
        self.assertTrue(os.path.isfile('RF1_sents.sv'))
        os.remove('RF1_sents.en')
        os.remove('RF1_sents.sv')
        os.remove('RF_latest_xml_en.zip')
        os.remove('RF_latest_xml_sv.zip')
        os.remove('RF_latest_xml_en-sv.xml.gz')

    @mock.patch('opustools.opus_get.input', create=True)
    def test_create_output_dir_if_it_does_not_exist(self, mocked_input):
        mocked_input.side_effect = ['y']
        common = {'output_directory': 'test_creating_dir'}
        step = self.configuration['steps'][0]
        test_config = {'common': common, 'steps': [step]}
        test_filter = OpusFilter(test_config)
        test_filter.execute_steps()
        self.assertTrue(os.path.isfile('test_creating_dir/RF1_sents.en'))
        self.assertTrue(os.path.isfile('test_creating_dir/RF1_sents.sv'))
        shutil.rmtree('test_creating_dir')


class TestExtraKeyErrors(unittest.TestCase):

    def test_read_from_opus(self):
        opusfilter = OpusFilter(
            {'steps': [
                {'type': 'opus_read', 'parameters': {
                    'corpus_name': 'RF', 'source_language': 'en', 'target_language': 'sv', 'relase': 'latest',
                    'preprocessing': 'xml', 'src_output': 'RF1_sents.en', 'tgt_output': 'RF1_sents.sv'}}
            ]})
        with self.assertRaises(ConfigurationError):
            opusfilter.execute_steps()

    def test_concatenate(self):
        opusfilter = OpusFilter(
            {'steps': [{'type': 'concatenate', 'parameters': {'inputs': [], 'outputs': []}}]})
        with self.assertRaises(ConfigurationError):
            opusfilter.execute_steps()

    def test_train_alignment(self):
        opusfilter = OpusFilter(
            {'steps': [{'type': 'train_alignment', 'parameters': {
                'src_data': 'foo', 'tgt_data': 'foo', 'parameters': {'mode': 3}}}]})
        with self.assertRaises(ConfigurationError):
            opusfilter.execute_steps()


class TestSort(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.opus_filter = OpusFilter({'common': {'output_directory': self.tempdir}, 'steps': []})
        with open(os.path.join(self.tempdir, 'rank_input_src'), 'w') as f:
            f.write('Sentence3\nSentence4\nSentence2\nSentence1')
        with open(os.path.join(self.tempdir, 'rank_input_tgt'), 'w') as f:
            f.write('Sentence3\nSentence4\nSentence2\nSentence1')
        with open(os.path.join(self.tempdir, 'ranks_input'), 'w') as f:
            f.write('0.5\n0\n2\n10')
        with open(os.path.join(self.tempdir, 'scores_input'), 'w') as f:
            for item in [{'MyScore': {'src': 1, 'tgt': 0.5}},
                         {'MyScore': {'src': 0.8, 'tgt': 0}},
                         {'MyScore': {'src': 0.5, 'tgt': 2}},
                         {'MyScore': {'src': 1, 'tgt': 10}}]:
                f.write(json.dumps(item) + '\n')
        with open(os.path.join(self.tempdir, 'scores_list_input'), 'w') as f:
            for item in [{'MyScore': [1, 0.5]},
                         {'MyScore': [0.8, 0]},
                         {'MyScore': [0.5, 2]},
                         {'MyScore': [1, 10]}]:
                f.write(json.dumps(item) + '\n')

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_order_by_rank(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'rank_input_src'),
                       os.path.join(self.tempdir, 'rank_input_tgt'),
                       os.path.join(self.tempdir, 'ranks_input')],
            'values': os.path.join(self.tempdir, 'ranks_input'),
            'outputs': [os.path.join(self.tempdir, 'rank_output_src'),
                        os.path.join(self.tempdir, 'rank_output_tgt'),
                        os.path.join(self.tempdir, 'ranks_output')],
            'reverse': False}
        self.opus_filter.sort_files(parameters)
        with open(os.path.join(self.tempdir, 'rank_output_src')) as f:
            self.assertEqual(f.read(), 'Sentence4\nSentence3\nSentence2\nSentence1\n')
        with open(os.path.join(self.tempdir, 'rank_output_tgt')) as f:
            self.assertEqual(f.read(), 'Sentence4\nSentence3\nSentence2\nSentence1\n')
        with open(os.path.join(self.tempdir, 'ranks_output')) as f:
            self.assertEqual(f.read(), '0\n0.5\n2\n10\n')

    def test_sort_files_reverse(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'rank_input_src'),
                       os.path.join(self.tempdir, 'rank_input_tgt'),
                       os.path.join(self.tempdir, 'ranks_input')],
            'values': os.path.join(self.tempdir, 'ranks_input'),
            'outputs': [os.path.join(self.tempdir, 'rank_output_src'),
                        os.path.join(self.tempdir, 'rank_output_tgt'),
                        os.path.join(self.tempdir, 'ranks_output')],
            'reverse': True}
        self.opus_filter.sort_files(parameters)
        with open(os.path.join(self.tempdir, 'rank_output_src')) as f:
            self.assertEqual(f.read(), 'Sentence1\nSentence2\nSentence3\nSentence4\n')
        with open(os.path.join(self.tempdir, 'rank_output_tgt')) as f:
            self.assertEqual(f.read(), 'Sentence1\nSentence2\nSentence3\nSentence4\n')
        with open(os.path.join(self.tempdir, 'ranks_output')) as f:
            self.assertEqual(f.read(), '10\n2\n0.5\n0\n')

    def test_sort_by_score(self):
        parameters1 = {
            'inputs': [os.path.join(self.tempdir, 'rank_input_src'),
                       os.path.join(self.tempdir, 'rank_input_tgt'),
                       os.path.join(self.tempdir, 'ranks_input')],
            'values': os.path.join(self.tempdir, 'scores_input'),
            'outputs': [os.path.join(self.tempdir, 'rank_output_src'),
                        os.path.join(self.tempdir, 'rank_output_tgt'),
                        os.path.join(self.tempdir, 'ranks_output')],
            'reverse': False,
            'key': 'MyScore.tgt'}
        parameters2 = copy.deepcopy(parameters1)
        parameters2['values'] = os.path.join(self.tempdir, 'scores_list_input')
        parameters2['key'] = 'MyScore.1'
        for parameters in [parameters1, parameters2]:
            self.opus_filter.sort_files(parameters)
            with open(os.path.join(self.tempdir, 'rank_output_src')) as f:
                self.assertEqual(f.read(), 'Sentence4\nSentence3\nSentence2\nSentence1\n')
            with open(os.path.join(self.tempdir, 'rank_output_tgt')) as f:
                self.assertEqual(f.read(), 'Sentence4\nSentence3\nSentence2\nSentence1\n')
            with open(os.path.join(self.tempdir, 'ranks_output')) as f:
                self.assertEqual(f.read(), '0\n0.5\n2\n10\n')

    def test_sort_by_str(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'rank_input_src'),
                       os.path.join(self.tempdir, 'rank_input_tgt'),
                       os.path.join(self.tempdir, 'ranks_input')],
            'values': os.path.join(self.tempdir, 'rank_input_src'),
            'outputs': [os.path.join(self.tempdir, 'rank_output_src'),
                        os.path.join(self.tempdir, 'rank_output_tgt'),
                        os.path.join(self.tempdir, 'ranks_output')],
            'reverse': False,
            'type': 'str'
        }
        self.opus_filter.sort_files(parameters)
        with open(os.path.join(self.tempdir, 'rank_output_src')) as f:
            self.assertEqual(f.read(), 'Sentence1\nSentence2\nSentence3\nSentence4\n')
        with open(os.path.join(self.tempdir, 'rank_output_tgt')) as f:
            self.assertEqual(f.read(), 'Sentence1\nSentence2\nSentence3\nSentence4\n')
        with open(os.path.join(self.tempdir, 'ranks_output')) as f:
            self.assertEqual(f.read(), '10\n2\n0.5\n0\n')


class TestJoin(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.opus_filter = OpusFilter({'common': {'output_directory': self.tempdir}, 'steps': []})
        with open(os.path.join(self.tempdir, 'scores_input'), 'w') as f:
            for item in [{'MyScore': {'src': 1, 'tgt': 0.5}, 'OtherScore': 0},
                         {'MyScore': {'src': 0.8, 'tgt': 0}, 'OtherScore': 0},
                         {'MyScore': {'src': 0.5, 'tgt': 2}, 'OtherScore': 0}]:
                f.write(json.dumps(item) + '\n')
        with open(os.path.join(self.tempdir, 'scores_input_2'), 'w') as f:
            for item in [{'OtherScore': 2}, {'OtherScore': 8}, {'OtherScore': 5}]:
                f.write(json.dumps(item) + '\n')
        with open(os.path.join(self.tempdir, 'ranks_input'), 'w') as f:
            f.write('0.5\n0\n2')

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_join_scores_flat(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'scores_input'),
                       os.path.join(self.tempdir, 'scores_input_2')],
            'output': os.path.join(self.tempdir, 'scores_output')}
        self.opus_filter.join_scores(parameters)
        with open(os.path.join(self.tempdir, 'scores_output')) as f:
            out = []
            for line in f:
                out.append(json.loads(line))
        self.assertSequenceEqual(out, [{'MyScore': {'src': 1, 'tgt': 0.5}, 'OtherScore': 2},
                                       {'MyScore': {'src': 0.8, 'tgt': 0}, 'OtherScore': 8},
                                       {'MyScore': {'src': 0.5, 'tgt': 2}, 'OtherScore': 5}])

    def test_join_scores_keys(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'scores_input'),
                       os.path.join(self.tempdir, 'scores_input_2')],
            'output': os.path.join(self.tempdir, 'scores_output'),
            'keys': [None, 'others']}
        self.opus_filter.join_scores(parameters)
        with open(os.path.join(self.tempdir, 'scores_output')) as f:
            out = []
            for line in f:
                out.append(json.loads(line))
        self.assertSequenceEqual(out, [
            {'MyScore': {'src': 1, 'tgt': 0.5}, 'OtherScore': 0, 'others': {'OtherScore': 2}},
            {'MyScore': {'src': 0.8, 'tgt': 0}, 'OtherScore': 0, 'others': {'OtherScore': 8}},
            {'MyScore': {'src': 0.5, 'tgt': 2}, 'OtherScore': 0, 'others': {'OtherScore': 5}}
        ])

    def test_join_scores_append(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'scores_input'),
                       os.path.join(self.tempdir, 'ranks_input')],
            'output': os.path.join(self.tempdir, 'scores_output'),
            'keys': [None, 'MyScore.value']}
        self.opus_filter.join_scores(parameters)
        with open(os.path.join(self.tempdir, 'scores_output')) as f:
            out = []
            for line in f:
                out.append(json.loads(line))
        self.assertSequenceEqual(out, [
            {'MyScore': {'src': 1, 'tgt': 0.5, 'value': 0.5}, 'OtherScore': 0},
            {'MyScore': {'src': 0.8, 'tgt': 0, 'value': 0}, 'OtherScore': 0},
            {'MyScore': {'src': 0.5, 'tgt': 2, 'value': 2}, 'OtherScore': 0}
        ])

    def test_join_scores_plain(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'scores_input_2'),
                       os.path.join(self.tempdir, 'ranks_input')],
            'output': os.path.join(self.tempdir, 'scores_output'),
            'keys': [None, 'rank']}
        self.opus_filter.join_scores(parameters)
        with open(os.path.join(self.tempdir, 'scores_output')) as f:
            out = []
            for line in f:
                out.append(json.loads(line))
        self.assertSequenceEqual(out, [{'OtherScore': 2, 'rank': 0.5},
                                       {'OtherScore': 8, 'rank': 0},
                                       {'OtherScore': 5, 'rank': 2}])

    def test_join_scores_plain_multikey(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'scores_input_2'),
                       os.path.join(self.tempdir, 'ranks_input')],
            'output': os.path.join(self.tempdir, 'scores_output'),
            'keys': [None, 'misc.rank']}
        self.opus_filter.join_scores(parameters)
        with open(os.path.join(self.tempdir, 'scores_output')) as f:
            out = []
            for line in f:
                out.append(json.loads(line))
        self.assertSequenceEqual(out, [{'OtherScore': 2, 'misc': {'rank': 0.5}},
                                       {'OtherScore': 8, 'misc': {'rank': 0}},
                                       {'OtherScore': 5, 'misc': {'rank': 2}}])


class TestHeadTailSlice(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.opus_filter = OpusFilter(
            {'common': {'output_directory': self.tempdir}, 'steps': []})
        with open(os.path.join(self.tempdir, 'input_src'), 'w') as f:
            f.write('Sentence3\nSentence4\nSentence2\nSentence1\n')
        with open(os.path.join(self.tempdir, 'input_tgt'), 'w') as f:
            f.write('sentence3\nsentence4\nsentence2\nsentence1\n')

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_head(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')],
            'n': 2}
        self.opus_filter.head(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), 'Sentence3\nSentence4\n')
        with open(os.path.join(self.tempdir, 'output_tgt')) as f:
            self.assertEqual(f.read(), 'sentence3\nsentence4\n')

    def test_tail(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')],
            'n': 2}
        self.opus_filter.tail(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), 'Sentence2\nSentence1\n')
        with open(os.path.join(self.tempdir, 'output_tgt')) as f:
            self.assertEqual(f.read(), 'sentence2\nsentence1\n')

    def test_slice_head(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')],
            'stop': 2}
        self.opus_filter.slice(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), 'Sentence3\nSentence4\n')
        with open(os.path.join(self.tempdir, 'output_tgt')) as f:
            self.assertEqual(f.read(), 'sentence3\nsentence4\n')

    def test_slice(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')],
            'start': 1, 'stop': None, 'step': 2}
        self.opus_filter.slice(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), 'Sentence4\nSentence1\n')
        with open(os.path.join(self.tempdir, 'output_tgt')) as f:
            self.assertEqual(f.read(), 'sentence4\nsentence1\n')


class TestProduct(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.opus_filter = OpusFilter(
            {'common': {'output_directory': self.tempdir}, 'steps': []})
        with open(os.path.join(self.tempdir, 'input_a_1'), 'w') as f:
            f.write('\n'.join(['a', 'o', 'u', 'i']) + '\n')
        with open(os.path.join(self.tempdir, 'input_a_2'), 'w') as f:
            f.write('\n'.join(['A', 'O', 'U', '']) + '\n')
        with open(os.path.join(self.tempdir, 'input_a_3'), 'w') as f:
            f.write('\n'.join(['ä', 'ö', '', '']) + '\n')
        with open(os.path.join(self.tempdir, 'input_b_1'), 'w') as f:
            f.write('\n'.join(['1', '2', '3', '']) + '\n')
        with open(os.path.join(self.tempdir, 'input_b_2'), 'w') as f:
            f.write('\n'.join(['I', 'II', '3', '']) + '\n')
        with open(os.path.join(self.tempdir, 'input_c_1'), 'w') as f:
            f.write('\n'.join(['-', '|', '+', 'x']) + '\n')

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_all(self):
        parameters = {
            'inputs': [[os.path.join(self.tempdir, 'input_a_1'),
                        os.path.join(self.tempdir, 'input_a_2'),
                        os.path.join(self.tempdir, 'input_a_3')],
                       [os.path.join(self.tempdir, 'input_b_1'),
                        os.path.join(self.tempdir, 'input_b_2')],
                       [os.path.join(self.tempdir, 'input_c_1')]],
            'outputs': [os.path.join(self.tempdir, 'output_a'),
                        os.path.join(self.tempdir, 'output_b'),
                        os.path.join(self.tempdir, 'output_c')],
            'skip_empty': False,
            'skip_duplicates': False,
            'k': None}
        self.opus_filter.product(parameters)
        with open(os.path.join(self.tempdir, 'output_a')) as f:
            self.assertEqual(
                f.read(), '\n'.join(
                    ['a', 'a', 'A', 'A', 'ä', 'ä',
                     'o', 'o', 'O', 'O', 'ö', 'ö',
                     'u', 'u', 'U', 'U', '', '',
                     'i', 'i', '', '', '', ''],
                ) + '\n')
        with open(os.path.join(self.tempdir, 'output_b')) as f:
            self.assertEqual(
                f.read(), '\n'.join(
                    ['1', 'I', '1', 'I', '1', 'I',
                     '2', 'II', '2', 'II', '2', 'II',
                     '3', '3', '3', '3', '3', '3',
                     '', '', '', '', '', ''],
                ) + '\n')
        with open(os.path.join(self.tempdir, 'output_c')) as f:
            self.assertEqual(
                f.read(), '\n'.join(
                    ['-', '-', '-', '-', '-', '-',
                     '|', '|', '|', '|', '|', '|',
                     '+', '+', '+', '+', '+', '+',
                     'x', 'x', 'x', 'x', 'x', 'x'],
                ) + '\n')

    def test_skip_empty(self):
        parameters = {
            'inputs': [[os.path.join(self.tempdir, 'input_a_1'),
                        os.path.join(self.tempdir, 'input_a_2'),
                        os.path.join(self.tempdir, 'input_a_3')],
                       [os.path.join(self.tempdir, 'input_b_1'),
                        os.path.join(self.tempdir, 'input_b_2')],
                       [os.path.join(self.tempdir, 'input_c_1')]],
            'outputs': [os.path.join(self.tempdir, 'output_a'),
                        os.path.join(self.tempdir, 'output_b'),
                        os.path.join(self.tempdir, 'output_c')],
            'skip_empty': True,
            'skip_duplicates': False,
            'k': None}
        self.opus_filter.product(parameters)
        with open(os.path.join(self.tempdir, 'output_a')) as f:
            self.assertEqual(
                f.read(), '\n'.join(
                    ['a', 'a', 'A', 'A', 'ä', 'ä',
                     'o', 'o', 'O', 'O', 'ö', 'ö',
                     'u', 'u', 'U', 'U'],
                ) + '\n')
        with open(os.path.join(self.tempdir, 'output_b')) as f:
            self.assertEqual(
                f.read(), '\n'.join(
                    ['1', 'I', '1', 'I', '1', 'I',
                     '2', 'II', '2', 'II', '2', 'II',
                     '3', '3', '3', '3'],
                ) + '\n')
        with open(os.path.join(self.tempdir, 'output_c')) as f:
            self.assertEqual(
                f.read(), '\n'.join(
                    ['-', '-', '-', '-', '-', '-',
                     '|', '|', '|', '|', '|', '|',
                     '+', '+', '+', '+'],
                ) + '\n')

    def test_skip_duplicates(self):
        parameters = {
            'inputs': [[os.path.join(self.tempdir, 'input_a_1'),
                        os.path.join(self.tempdir, 'input_a_2'),
                        os.path.join(self.tempdir, 'input_a_3')],
                       [os.path.join(self.tempdir, 'input_b_1'),
                        os.path.join(self.tempdir, 'input_b_2')],
                       [os.path.join(self.tempdir, 'input_c_1')]],
            'outputs': [os.path.join(self.tempdir, 'output_a'),
                        os.path.join(self.tempdir, 'output_b'),
                        os.path.join(self.tempdir, 'output_c')],
            'skip_empty': True,
            'skip_duplicates': True,
            'k': None}
        self.opus_filter.product(parameters)
        with open(os.path.join(self.tempdir, 'output_a')) as f:
            self.assertEqual(
                f.read(), '\n'.join(
                    ['A', 'A', 'a', 'a', 'ä', 'ä',
                     'O', 'O', 'o', 'o', 'ö', 'ö',
                     'U', 'u'],  # Note: sorted order
                ) + '\n')
        with open(os.path.join(self.tempdir, 'output_b')) as f:
            self.assertEqual(
                f.read(), '\n'.join(
                    ['1', 'I', '1', 'I', '1', 'I',
                     '2', 'II', '2', 'II', '2', 'II',
                     '3', '3'],
                ) + '\n')
        with open(os.path.join(self.tempdir, 'output_c')) as f:
            self.assertEqual(
                f.read(), '\n'.join(
                    ['-', '-', '-', '-', '-', '-',
                     '|', '|', '|', '|', '|', '|',
                     '+', '+'],
                ) + '\n')

    def test_sample(self):
        parameters = {
            'inputs': [[os.path.join(self.tempdir, 'input_a_1'),
                        os.path.join(self.tempdir, 'input_a_2'),
                        os.path.join(self.tempdir, 'input_a_3')],
                       [os.path.join(self.tempdir, 'input_b_1'),
                        os.path.join(self.tempdir, 'input_b_2')],
                       [os.path.join(self.tempdir, 'input_c_1')]],
            'outputs': [os.path.join(self.tempdir, 'output_a'),
                        os.path.join(self.tempdir, 'output_b'),
                        os.path.join(self.tempdir, 'output_c')],
            'skip_empty': True,
            'skip_duplicates': True,
            'k': 3}
        self.opus_filter.product(parameters)
        with open(os.path.join(self.tempdir, 'output_a')) as f:
            self.assertEqual(len(f.read().strip().split('\n')), 8)  # 2 x 3 + 2
        with open(os.path.join(self.tempdir, 'output_b')) as f:
            self.assertEqual(len(f.read().strip().split('\n')), 8)  # 2 x 3 + 2
        with open(os.path.join(self.tempdir, 'output_c')) as f:
            self.assertEqual(len(f.read().strip().split('\n')), 8)  # 2 x 3 + 2


class TestSubset(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.opus_filter = OpusFilter(
            {'common': {'output_directory': self.tempdir}, 'steps': []})
        with open(os.path.join(self.tempdir, 'input_src'), 'w') as f:
            f.write(''.join('sent_{}\n'.format(idx) for idx in range(100)))
        with open(os.path.join(self.tempdir, 'input_tgt'), 'w') as f:
            f.write(''.join('sent_{}\n'.format(idx) for idx in range(100)))

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_subset(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')],
            'size': 50}
        self.opus_filter.get_subset(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as fobj1, \
             open(os.path.join(self.tempdir, 'output_tgt')) as fobj2:
            lines1 = fobj1.readlines()
            lines2 = fobj2.readlines()
            self.assertEqual(len(lines1), 50)
            self.assertEqual(len(lines2), 50)
            self.assertSequenceEqual(lines1, lines2)

    def test_subset_shuffle(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')],
            'size': 20, 'shuffle_subset': True, 'seed': 1223}
        self.opus_filter.get_subset(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as fobj1, \
             open(os.path.join(self.tempdir, 'output_tgt')) as fobj2:
            lines1 = fobj1.readlines()
            lines2 = fobj2.readlines()
            self.assertEqual(len(lines1), 20)
            self.assertEqual(len(lines2), 20)
            self.assertFalse(all(l1 == l2 for l1, l2 in zip(lines1, lines2)))


class TestSplit(unittest.TestCase):

    # TODO: Replace with tests that are do not depend on the specific
    # split with the current data and hash algorithm.

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.opus_filter = OpusFilter(
            {'common': {'output_directory': self.tempdir}, 'steps': []})
        with open(os.path.join(self.tempdir, 'input_src'), 'w') as f:
            f.write(''.join('Sent_{}\n'.format(idx) for idx in range(6)))
        with open(os.path.join(self.tempdir, 'input_tgt'), 'w') as f:
            f.write(''.join('sent_{}\n'.format(idx) for idx in range(6)))

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_split_single_out(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')],
            'divisor': 2, 'hash': 'xx_64'}
        self.opus_filter.split(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), ''.join('Sent_{}\n'.format(idx) for idx in [1, 2, 5]))
        with open(os.path.join(self.tempdir, 'output_tgt')) as f:
            self.assertEqual(f.read(), ''.join('sent_{}\n'.format(idx) for idx in [1, 2, 5]))

    def test_split_single_out_seed(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')],
            'divisor': 2, 'hash': 'xx_64', 'seed': 123}
        self.opus_filter.split(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), ''.join('Sent_{}\n'.format(idx) for idx in [1]))
        with open(os.path.join(self.tempdir, 'output_tgt')) as f:
            self.assertEqual(f.read(), ''.join('sent_{}\n'.format(idx) for idx in [1]))

    def test_split_two_out(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')],
            'outputs_2': [os.path.join(self.tempdir, 'output_src_2'),
                          os.path.join(self.tempdir, 'output_tgt_2')],
            'divisor': 2, 'hash': 'xx_64'}
        self.opus_filter.split(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), ''.join('Sent_{}\n'.format(idx) for idx in [1, 2, 5]))
        with open(os.path.join(self.tempdir, 'output_tgt')) as f:
            self.assertEqual(f.read(), ''.join('sent_{}\n'.format(idx) for idx in [1, 2, 5]))

        with open(os.path.join(self.tempdir, 'output_src_2')) as f:
            self.assertEqual(f.read(), ''.join('Sent_{}\n'.format(idx) for idx in [0, 3, 4]))
        with open(os.path.join(self.tempdir, 'output_tgt_2')) as f:
            self.assertEqual(f.read(), ''.join('sent_{}\n'.format(idx) for idx in [0, 3, 4]))

    def test_split_src_key(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')],
            'divisor': 2, 'compare': [0], 'hash': 'xx_64'}
        self.opus_filter.split(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), ''.join('Sent_{}\n'.format(idx) for idx in [3, 4]))
        with open(os.path.join(self.tempdir, 'output_tgt')) as f:
            self.assertEqual(f.read(), ''.join('sent_{}\n'.format(idx) for idx in [3, 4]))

    def test_split_tgt_key(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')],
            'divisor': 2, 'compare': [1], 'hash': 'xx_64'}
        self.opus_filter.split(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), ''.join('Sent_{}\n'.format(idx) for idx in [2, 4, 5]))
        with open(os.path.join(self.tempdir, 'output_tgt')) as f:
            self.assertEqual(f.read(), ''.join('sent_{}\n'.format(idx) for idx in [2, 4, 5]))

    def test_split_modulo_threshold(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')],
            'divisor': 10, 'threshold': 3, 'hash': 'xx_64'}
        self.opus_filter.split(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), ''.join('Sent_{}\n'.format(idx) for idx in [1, 3, 5]))
        with open(os.path.join(self.tempdir, 'output_tgt')) as f:
            self.assertEqual(f.read(), ''.join('sent_{}\n'.format(idx) for idx in [1, 3, 5]))


class TestRemoveDuplicates(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.opus_filter = OpusFilter(
            {'common': {'output_directory': self.tempdir}, 'steps': []})
        with open(os.path.join(self.tempdir, 'input_src'), 'w') as f:
            f.write('\n'.join(['a', 'b', 'c', 'd', 'e', 'a', 'b', 'b', 'f']) + '\n')
        with open(os.path.join(self.tempdir, 'input_tgt'), 'w') as f:
            f.write('\n'.join(['A', 'B', 'C', 'D', 'E', 'A', 'B', 'F', 'C']) + '\n')

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_defaults(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')]}
        self.opus_filter.remove_duplicates(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), 'a\nb\nc\nd\ne\nb\nf\n')
        with open(os.path.join(self.tempdir, 'output_tgt')) as f:
            self.assertEqual(f.read(), 'A\nB\nC\nD\nE\nF\nC\n')

    def test_nohash(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')],
            'hash': None}
        self.opus_filter.remove_duplicates(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), 'a\nb\nc\nd\ne\nb\nf\n')
        with open(os.path.join(self.tempdir, 'output_tgt')) as f:
            self.assertEqual(f.read(), 'A\nB\nC\nD\nE\nF\nC\n')

    def test_src_key_only(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')],
            'compare': [0]}
        self.opus_filter.remove_duplicates(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), 'a\nb\nc\nd\ne\nf\n')
        with open(os.path.join(self.tempdir, 'output_tgt')) as f:
            self.assertEqual(f.read(), 'A\nB\nC\nD\nE\nC\n')

    def test_tgt_key_only(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')],
            'compare': [1]}
        self.opus_filter.remove_duplicates(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), 'a\nb\nc\nd\ne\nb\n')
        with open(os.path.join(self.tempdir, 'output_tgt')) as f:
            self.assertEqual(f.read(), 'A\nB\nC\nD\nE\nF\n')

    def test_single_file(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src')],
            'outputs': [os.path.join(self.tempdir, 'output_src')]
        }
        self.opus_filter.remove_duplicates(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), 'a\nb\nc\nd\ne\nf\n')


class TestRemoveDuplicatesPreprocess(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.opus_filter = OpusFilter(
            {'common': {'output_directory': self.tempdir}, 'steps': []})
        with open(os.path.join(self.tempdir, 'input_src'), 'w') as f:
            f.write('\n'.join(['a', 'b', 'c', 'B', 'b?', 'B!']) + '\n')
        with open(os.path.join(self.tempdir, 'input_tgt'), 'w') as f:
            f.write('\n'.join(['A', 'B', 'C', 'B', 'B', 'B']) + '\n')

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_defaults(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')]}
        self.opus_filter.remove_duplicates(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), 'a\nb\nc\nB\nb?\nB!\n')
        with open(os.path.join(self.tempdir, 'output_tgt')) as f:
            self.assertEqual(f.read(), 'A\nB\nC\nB\nB\nB\n')

    def test_preprocessed(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')],
            'letters_only': True,
            'lowercase': True
        }
        self.opus_filter.remove_duplicates(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), 'a\nb\nc\n')
        with open(os.path.join(self.tempdir, 'output_tgt')) as f:
            self.assertEqual(f.read(), 'A\nB\nC\n')


class TestRemoveDuplicatesOverlap(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.opus_filter = OpusFilter(
            {'common': {'output_directory': self.tempdir}, 'steps': []})
        with open(os.path.join(self.tempdir, 'input_src'), 'w') as f:
            f.write('\n'.join(['a', 'b', 'c', 'd', 'e', 'a', 'b', 'b', 'f']) + '\n')
        with open(os.path.join(self.tempdir, 'input_tgt'), 'w') as f:
            f.write('\n'.join(['A', 'B', 'C', 'D', 'E', 'A', 'B', 'F', 'C']) + '\n')
        with open(os.path.join(self.tempdir, 'overlap_src'), 'w') as f:
            f.write('\n'.join(['b', 'd', 'e']) + '\n')
        with open(os.path.join(self.tempdir, 'overlap_tgt'), 'w') as f:
            f.write('\n'.join(['B', 'D', 'E']) + '\n')

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_defaults(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')],
            'overlap': [os.path.join(self.tempdir, 'overlap_src'),
                        os.path.join(self.tempdir, 'overlap_tgt')]}
        self.opus_filter.remove_duplicates(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), 'a\nc\na\nb\nf\n')
        with open(os.path.join(self.tempdir, 'output_tgt')) as f:
            self.assertEqual(f.read(), 'A\nC\nA\nF\nC\n')


class TestUnzip(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.opus_filter = OpusFilter(
            {'common': {'output_directory': self.tempdir}, 'steps': []})
        with open(os.path.join(self.tempdir, 'input'), 'w') as f:
            f.write('Sentence1\tsentence1\nSentence2\tsentence2\n')

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_head(self):
        parameters = {
            'input': os.path.join(self.tempdir, 'input'),
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')],
            'separator': '\t'}
        self.opus_filter.unzip(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), 'Sentence1\nSentence2\n')
        with open(os.path.join(self.tempdir, 'output_tgt')) as f:
            self.assertEqual(f.read(), 'sentence1\nsentence2\n')


class TestPreprocess(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.opus_filter = OpusFilter(
            {'common': {'output_directory': self.tempdir}, 'steps': []})
        self.inputs = [
            ["Hello, world!", "(1) Punctuation, e.g., comma", "(2) C. done 4.5"],
            ["Hei, maailma!", "(1) Välimerkit, esim. pilkku", "(2) C. valmis 4,5"]
        ]
        self.expected = [
            ["Hello , world !", "Punctuation , e.g. , comma", "C. done 4.5"],
            ["Hei , maailma !", "Välimerkit , esim. pilkku", "C. valmis 4,5"]
        ]
        with open(os.path.join(self.tempdir, 'input_src'), 'w') as f:
            f.write('\n'.join(self.inputs[0]))
        with open(os.path.join(self.tempdir, 'input_tgt'), 'w') as f:
            f.write('\n'.join(self.inputs[1]))

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_preprocess(self):
        parameters = {
            'inputs': [os.path.join(self.tempdir, 'input_src'),
                       os.path.join(self.tempdir, 'input_tgt')],
            'outputs': [os.path.join(self.tempdir, 'output_src'),
                        os.path.join(self.tempdir, 'output_tgt')],
            'preprocessors': [
                {'WhitespaceNormalizer': {}},
                {'RegExpSub': {'patterns': ([(r"^ *\([0-9-]+\) *", "", 0, [])])}},
                {'Tokenizer': {'tokenizer': 'moses', 'languages': ['fi', 'en']}},
            ]}
        self.opus_filter.preprocess(parameters)
        with open(os.path.join(self.tempdir, 'output_src')) as f:
            self.assertEqual(f.read(), '\n'.join(self.expected[0]) + '\n')
        with open(os.path.join(self.tempdir, 'output_tgt')) as f:
            self.assertEqual(f.read(), '\n'.join(self.expected[1]) + '\n')


class TestDownload(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.opus_filter = OpusFilter(
            {'common': {'output_directory': self.tempdir}, 'steps': []})

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_download(self):
        parameters = {'output': os.path.join(self.tempdir, 'output'),
                      'url': 'https://github.com/Helsinki-NLP/OpusFilter/raw/master/README.md'}
        try:
            self.opus_filter.download_file(parameters)
        except requests.exceptions.ConnectionError:
            self.skipTest("Failed to download test resources")
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir, 'output')))


class TestWrite(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.opus_filter = OpusFilter(
            {'common': {'output_directory': self.tempdir}, 'steps': []})

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_write(self):
        data = 'foo\nbar\n99\n'
        parameters = {'output': os.path.join(self.tempdir, 'output'), 'data': data}
        self.opus_filter.write_to_file(parameters)
        with open(os.path.join(self.tempdir, 'output')) as f:
            self.assertEqual(f.read(), str(data))


class TestVariables(unittest.TestCase):

    def setUp(self):
        self.of = OpusFilter({'steps': []})

    def test_check_variables_valid(self):
        self.assertEqual(self.of._check_variables({}), 0)
        self.assertEqual(self.of._check_variables({'key': []}), 0)
        self.assertEqual(self.of._check_variables({'key': [1]}), 1)
        self.assertEqual(self.of._check_variables({'key': ['a', 'b', 'c']}), 3)
        self.assertEqual(self.of._check_variables({'key1': ['a', 'b', 'c'], 'key2': [1, 2, 3]}), 3)

    def test_check_variables_invalid(self):
        with self.assertRaises(ConfigurationError):
            self.of._check_variables({'key': 1})
        with self.assertRaises(ConfigurationError):
            self.of._check_variables({'key': {}})
        with self.assertRaises(ConfigurationError):
            self.of._check_variables({'key1': ['a', 'b', 'c'], 'key2': [1, 2]})

    def test_expand_parameters_empty(self):
        for case in [
                None, 'abc', 10, [], ['a'], {}, {'a': 3}, [{'foo': [1, 2]}]
        ]:
            self.assertEqual(self.of._expand_parameters(case, {}), case)

    def test_expand_parameters(self):
        variables = {'myint': 5, 'mylist': ['a', 'b'], 'mystr': 'bar'}
        for case, expected in [
                (Var('mystr'), variables['mystr']),
                (Var('myint'), variables['myint']),
                (Var('mylist'), variables['mylist']),
                (VarStr('f_{mystr}.txt'), 'f_{mystr}.txt'.format(**variables)),
                ([Var('myint')], [variables['myint']]),
                ([VarStr('f_{mystr}.txt')], ['f_{mystr}.txt'.format(**variables)]),
                ({'key': Var('myint')}, {'key': variables['myint']}),
                ([{'key': Var('myint'), 'other': 0}, Var('mylist'), [VarStr('f_{mystr}.txt')]],
                 [{'key': variables['myint'], 'other': 0}, variables['mylist'],
                  ['f_{mystr}.txt'.format(**variables)]])
        ]:
            self.assertEqual(self.of._expand_parameters(case, variables), expected)

    def test_expand_parameters_invalid(self):
        variables = {'myint': 5, 'mylist': ['a', 'b'], 'mystr': 'bar'}
        for case in [Var('unk'), VarStr('{}'), VarStr('{unk}'), VarStr('{mystr}-{unk}')]:
            with self.assertRaises(ConfigurationError):
                self.of._expand_parameters(case, variables)


@unittest.skipIf('varikn' not in globals() or os.environ.get('EFLOMAL_PATH') is None, 'varikn or eflomal not found')
class TestParallel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tempdir = tempfile.mkdtemp()
        self.configuration = {
            'common': {'output_directory': self.tempdir},
            'steps':
            [{'type': 'opus_read',
              'parameters': {'corpus_name': 'RF',
                             'source_language': 'en',
                             'target_language': 'sv',
                             'release': 'latest',
                             'preprocessing': 'xml',
                             'src_output': 'RF1_sents.en',
                             'tgt_output': 'RF1_sents.sv'}},
             {'type': 'preprocess',
              'parameters': {
                    'inputs': ['RF1_sents.en', 'RF1_sents.sv'],
                    'outputs': ['RF1_preprocessed.en', 'RF1_preprocessed.sv'],
                    'n_jobs': 5,
                    'preprocessors': [{'Tokenizer':
                                      {'languages': ['en', 'sv'],
                                       'tokenizer': 'moses'}}]}},
             {'type': 'filter',
              'parameters': {
                  'inputs': ['RF1_preprocessed.en', 'RF1_preprocessed.sv'],
                  'outputs': ['RF1_filtered.en', 'RF1_filtered.sv'],
                  'n_jobs': 5,
                  'filters': [{'LanguageIDFilter':
                               {'languages': ['en', 'sv'],
                                'thresholds': [0, 0]}},
                              {'TerminalPunctuationFilter':
                               {'threshold': -2}},
                              {'NonZeroNumeralsFilter': {'threshold': 0.5}},
                              {'CharacterScoreFilter':
                               {'scripts': ['Latin', 'Latin'],
                                'thresholds': [1, 1]}}]}},
             {'type': 'train_ngram',
              'parameters': {'data': 'RF1_filtered.en',
                             'parameters': {'norder': 20, 'dscale': 0.001},
                             'model': 'RF1_en.arpa'}},
             {'type': 'train_ngram',
              'parameters': {'data': 'RF1_filtered.sv',
                             'parameters': {'norder': 20, 'dscale': 0.001},
                             'model': 'RF1_sv.arpa'}},
             {'type': 'train_alignment',
              'parameters': {'src_data': 'RF1_filtered.en',
                             'tgt_data': 'RF1_filtered.sv',
                             'parameters': {'src_tokenizer': None, 'tgt_tokenizer': None, 'model': 3},
                             'output': 'RF1_align.priors'}},
             {'type': 'score',
              'parameters': {
                  'inputs': ['RF1_sents.en', 'RF1_sents.sv'],
                  'output': 'RF1_scores.en-sv.jsonl',
                  'n_jobs': 5,
                  'filters': [{'LanguageIDFilter':
                               {'languages': ['en', 'sv'],
                                'thresholds': [0, 0]}},
                              {'TerminalPunctuationFilter':
                               {'threshold': -2}},
                              {'NonZeroNumeralsFilter': {'threshold': 0.5}},
                              {'CharacterScoreFilter':
                               {'scripts': ['Latin', 'Latin'],
                                'sthreshold': [1, 1]}},
                              {'WordAlignFilter': {'priors': 'RF1_align.priors',
                                                   'model': 3,
                                                   'src_threshold': 0,
                                                   'tgt_threshold': 0}},
                              {'CrossEntropyFilter':
                               {'lm_params': [{'filename': 'RF1_en.arpa'},
                                              {'filename': 'RF1_sv.arpa'}],
                                'thresholds': [50.0, 50.0],
                                'diff_threshold': 10.0}}]}},
             ]}
        OpusGet(directory='RF', source='en', target='sv', release='latest',
                preprocess='xml', suppress_prompts=True, download_dir=self.tempdir
                ).get_files()
        self.opus_filter = OpusFilter(self.configuration)
        self.opus_filter.execute_steps()

    def test_parallel_preprocess(self):
        assert os.path.exists(os.path.join(self.tempdir, 'RF1_preprocessed.en'))
        assert os.path.exists(os.path.join(self.tempdir, 'RF1_preprocessed.sv'))
        assert count_lines(os.path.join(self.tempdir, 'RF1_preprocessed.en')) == \
            count_lines(os.path.join(self.tempdir, 'RF1_sents.en'))
        assert count_lines(os.path.join(self.tempdir, 'RF1_preprocessed.sv')) == \
            count_lines(os.path.join(self.tempdir, 'RF1_sents.sv'))

    def test_parallel_filter(self):
        assert os.path.exists(os.path.join(self.tempdir, 'RF1_filtered.en'))
        assert os.path.exists(os.path.join(self.tempdir, 'RF1_filtered.sv'))
        assert count_lines(os.path.join(self.tempdir, 'RF1_filtered.en')) < \
            count_lines(os.path.join(self.tempdir, 'RF1_preprocessed.en'))
        assert count_lines(os.path.join(self.tempdir, 'RF1_filtered.sv')) < \
            count_lines(os.path.join(self.tempdir, 'RF1_preprocessed.sv'))

    def test_parallel_score(self):
        assert os.path.exists(os.path.join(self.tempdir, 'RF1_scores.en-sv.jsonl'))
        assert count_lines(os.path.join(self.tempdir, 'RF1_scores.en-sv.jsonl')) == \
            count_lines(os.path.join(self.tempdir, 'RF1_sents.en'))


class TestParallelWrapper(unittest.TestCase):
    def setUp(self):
        self.parameters = [
            {"num_lines": 1, "n_jobs": 5, 'limit': None, 'format': None},  # Test edge condition， n_jobs greater than num_lines
            {"num_lines": 100, "n_jobs": 1, 'limit': None, 'format': None},
            {"num_lines": 200, "n_jobs": 9, 'limit': None, 'format': None},
            {"num_lines": 200, "n_jobs": 10, 'limit': None, 'format': ".gz"},  # Test gzip format inputs and outputs
            {"num_lines": 50, "n_jobs": 10, 'limit': 20, 'format': None},
        ]

    def test_split_merge(self):
        for param in self.parameters:
            format = param.get('format', None)
            inputs = [tempfile.mkstemp(suffix=format)[1], tempfile.mkstemp(suffix=format)[1]]
            outputs = [tempfile.mkstemp(suffix=format)[1], tempfile.mkstemp(suffix=format)[1]]

            for input_ in inputs:
                fin = file_open(input_, 'w')
                for i in range(param["num_lines"]):
                    fin.write("{}\n".format(i))
                fin.close()

            in_chunked_files, out_chunked_files = ParallelWrapper.split(inputs, outputs, param["n_jobs"])
            assert len(in_chunked_files) == min(param["n_jobs"], param["num_lines"])
            assert len(out_chunked_files) == min(param["n_jobs"], param["num_lines"])
            for files in in_chunked_files:
                num_lines = [count_lines(f) for f in files]
                assert all(n == num_lines[0] for n in num_lines)
            # just copy the files
            for in_files, out_files in zip(in_chunked_files, out_chunked_files):
                for fin, fout in zip(in_files, out_files):
                    shutil.copyfile(fin, fout)
            ParallelWrapper.merge(in_chunked_files, outputs, out_chunked_files, param.get("limit", None))
            for output in outputs:
                if param.get("limit", None) is not None:
                    assert count_lines(output) == param["limit"]
                else:
                    assert count_lines(output) == param["num_lines"]

    def test_parallelize(self):
        mock_obj = Namespace()
        mock_obj.output_dir = tempfile.mkdtemp()
        mock_obj.default_n_jobs = 1
        mock_obj._check_extra_parameters = OpusFilter._check_extra_parameters

        @ParallelWrapper({'inputs', 'outputs', 'limit'})
        def func(self, parameters, overwrite=False):
            inputs = parameters['inputs']
            outputs = parameters['outputs']
            for input_, output in zip(inputs, outputs):
                input_ = os.path.join(self.output_dir, input_)
                output = os.path.join(self.output_dir, output)
                shutil.copyfile(input_, output)

        for param in self.parameters:
            format = param.get('format', None)
            inputs = [tempfile.mkstemp(dir=mock_obj.output_dir, suffix=format)[1],
                      tempfile.mkstemp(dir=mock_obj.output_dir, suffix=format)[1]]
            rel_inputs = [os.path.basename(path) for path in inputs]
            outputs = [tempfile.mkstemp(dir=mock_obj.output_dir, suffix=format)[1],
                       tempfile.mkstemp(dir=mock_obj.output_dir, suffix=format)[1]]
            rel_outputs = [os.path.basename(path) for path in outputs]

            for input_ in inputs:
                fin = file_open(input_, 'w')
                for i in range(param["num_lines"]):
                    fin.write("{}\n".format(i))
                fin.close()
            n_jobs = param["n_jobs"]
            func(mock_obj, {'inputs': rel_inputs, 'outputs': rel_outputs, "n_jobs": n_jobs,
                            "limit": param.get('limit', None)}, overwrite=True)
            for output in outputs:
                if param.get("limit", None) is not None:
                    assert count_lines(output) == param["limit"]
                else:
                    assert count_lines(output) == param["num_lines"]
