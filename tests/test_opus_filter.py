import unittest
from unittest import mock
import json
import argparse
import os
import shutil
import tempfile

from opustools import OpusGet
from opusfilter.opusfilter import OpusFilter


class TestOpusFilter(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tempdir = tempfile.mkdtemp()
        self.configuration = {'common': {'output_directory': self.tempdir},
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
                    'parameters': {'src_input': 'RF1_sents.en',
                        'tgt_input': 'RF1_sents.sv',
                        'src_output': 'RF1_filtered.en',
                        'tgt_output': 'RF1_filtered.sv',
                        'filters': [{'LanguageIDFilter':
                            {'src_lang': 'en',
                                'tgt_lang': 'sv',
                                'src_threshold': 0,
                                'tgt_threshold': 0}},
                            {'TerminalPunctuationFilter':
                                {'threshold': -2}},
                            {'NonZeroNumeralsFilter': {'threshold': 0.5}},
                            {'CharacterScoreFilter':
                                {'src_script': 'Latin',
                                    'tgt_script': 'Latin',
                                    'src_threshold': 1,
                                    'tgt_threshold': 1}}]}},
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
                        'parameters': {'tokenizer': 'none', 'model': 3},
                        'output': 'RF1_align.priors'}},
                {'type': 'score',
                    'parameters': {'src_input': 'RF1_sents.en',
                        'tgt_input': 'RF1_sents.sv',
                        'output': 'RF1_scores.en-sv.jsonl',
                        'filters': [{'LanguageIDFilter':
                            {'src_lang': 'en',
                                'tgt_lang': 'sv',
                                'src_threshold': 0,
                                'tgt_threshold': 0}},
                            {'TerminalPunctuationFilter':
                                {'threshold': -2}},
                            {'NonZeroNumeralsFilter': {'threshold': 0.5}},
                            {'CharacterScoreFilter':
                                {'src_script': 'Latin',
                                    'tgt_script': 'Latin',
                                    'src_threshold': 1,
                                    'tgt_threshold': 1}},
                            {'WordAlignFilter': {'tokenizer': 'none',
                                'priors': 'RF1_align.priors',
                                'model': 3,
                                'src_threshold': 0,
                                'tgt_threshold': 0}},
                            {'CrossEntropyFilter':
                                {'src_lm_params': {'filename': 'RF1_en.arpa'},
                                'tgt_lm_params': {'filename': 'RF1_sv.arpa'},
                                'src_threshold': 50.0,
                                'tgt_threshold': 50.0,
                                'diff_threshold': 10.0}}]}}]}

        OpusGet(directory='RF', source='en', target='sv', release='latest',
            preprocess='xml', suppress_prompts=True, download_dir=self.tempdir
            ).get_files()
        self.opus_filter = OpusFilter(self.configuration)
        self.opus_filter.execute_steps()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.tempdir)

    def test_get_pairs(self):
        pair_gen = self.opus_filter.get_pairs('RF1_sents.en', 'RF1_sents.sv')
        pair = next(pair_gen)
        for pair in pair_gen:
            pass
        self.assertEqual(pair,
                ('This will ensure the cohesion of Swedish society .',
                'Så kan vi hålla samman Sverige .'))

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

    def test_score_data(self):
        with open(os.path.join(self.tempdir, 'RF1_scores.en-sv.jsonl')) as scores_file:
            score = json.loads(scores_file.readline())
            self.assertEqual(score['LanguageIDFilter'], {'src': 1.0, 'tgt': 0.98})
            self.assertEqual(score['LanguageIDFilter'], {'src': 1.0, 'tgt': 0.98})
            self.assertEqual(score['CharacterScoreFilter'], {'src': 1.0, 'tgt': 1.0})
            self.assertAlmostEqual(
                score['CrossEntropyFilter']['src'], 15.214258903317491)
            self.assertAlmostEqual(
                score['CrossEntropyFilter']['tgt'], 7.569084909162213)
            self.assertEqual(score['TerminalPunctuationFilter'], -0.0)
            self.assertEqual(score['NonZeroNumeralsFilter'], 0.0)
            self.assertEqual(type(score['WordAlignFilter']), dict)

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
