"""Command-line interface for opusfilter-slurm command."""
import argparse
import logging
import sys

from opusfilter.util import yaml
from opusfilter.slurm import SlurmOpusFilter

logger = logging.getLogger(__name__)


def main(args=None):
    """Main entry point for opusfilter-slurm command."""
    parser = argparse.ArgumentParser(prog='opusfilter-slurm',
        description='Run OpusFilter workflows on SLURM clusters')

    parser.add_argument('config', metavar='CONFIG', help='YAML configuration file')
    parser.add_argument('--overwrite', '-o',
        help='overwrite existing output files', action='store_true')
    parser.add_argument('--resume', '-r',
        help='resume from last completed step', action='store_true')
    parser.add_argument('--dry-run', '-n',
        help='show what would be done without submitting jobs', action='store_true')
    parser.add_argument('--max-concurrent', type=int, default=None,
        help='maximum number of concurrent jobs')
    parser.add_argument('--workdir',
        help='working directory for SLURM scripts and logs')
    parser.add_argument('--email',
        help='override email for notifications')

    args = parser.parse_args(args)

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('mosestokenizer.tokenizer.MosesTokenizer').setLevel(logging.WARNING)

    # Load configuration
    configuration = yaml.load(open(args.config))
    configuration['_config_file'] = args.config  # Store for reference

    # Create and run SlurmOpusFilter
    slurm_filter = SlurmOpusFilter(configuration,
        workdir=args.workdir,
        max_concurrent=args.max_concurrent,
        email=args.email,
        dry_run=args.dry_run)

    slurm_filter.run(overwrite=args.overwrite, resume=args.resume)

    return 0


if __name__ == '__main__':
    sys.exit(main())
