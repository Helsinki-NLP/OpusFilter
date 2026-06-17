"""Command-line interface for OpusFilter main command."""
import argparse
import logging
import sys

from opusfilter.opusfilter import OpusFilter
from opusfilter.util import yaml, expand_single_step


def _silence_tqdm():
    """Disable tqdm when stderr is not a TTY (e.g. SLURM log files)."""
    if not sys.stderr.isatty():
        from tqdm import tqdm as _tqdm
        _orig_init = _tqdm.__init__
        def _init(self, *args, **kwargs):
            kwargs.setdefault('disable', True)
            return _orig_init(self, *args, **kwargs)
        _tqdm.__init__ = _init


def main(args=None):
    """Main entry point for opusfilter command."""
    parser = argparse.ArgumentParser(prog='opusfilter',
        description='Filter OPUS bitexts')

    parser.add_argument('config', metavar='CONFIG', help='YAML configuration file')
    parser.add_argument('--overwrite', '-o', help='overwrite existing output files', action='store_true')
    parser.add_argument('--last', type=int, default=None, help='Last step to run')
    parser.add_argument('--single', type=int, default=None, help='Run only the nth step')
    parser.add_argument('--substep', type=int, default=None,
        help='Run a specific substep (requires --single). '
             'Expands variables for the Nth variant (1-based index).')
    parser.add_argument('--n-jobs', type=int, default=None,
        help='Number of parallel jobs when running score, filter and preprocess.')

    args = parser.parse_args(args)

    _silence_tqdm()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('mosestokenizer.tokenizer.MosesTokenizer').setLevel(logging.WARNING)

    configuration = yaml.load(open(args.config))
    if args.n_jobs is not None:
        configuration['common']['default_n_jobs'] = args.n_jobs

    if args.single is not None and args.substep is not None:
        step_index = args.single - 1
        substep_index = args.substep - 1  # Convert to 0-based
        if 0 <= step_index < len(configuration['steps']):
            common_constants = configuration.get('common', {}).get('constants', {})
            expanded_step = expand_single_step(
                configuration['steps'][step_index],
                substep_index,
                common_constants
            )
            configuration['steps'][step_index] = expanded_step
        of = OpusFilter(configuration)
        of.execute_step(args.single, overwrite=args.overwrite)
    else:
        of = OpusFilter(configuration)
        if args.single is None:
            of.execute_steps(overwrite=args.overwrite, last=args.last)
        else:
            of.execute_step(args.single, overwrite=args.overwrite)

    return 0


if __name__ == '__main__':
    sys.exit(main())
