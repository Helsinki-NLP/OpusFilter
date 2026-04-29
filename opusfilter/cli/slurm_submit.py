"""Command-line interface for opusfilter-slurm-submit command."""
import argparse
import logging
import sys
import os

from opusfilter.util import yaml
from opusfilter.slurm import SlurmOpusFilter

logger = logging.getLogger(__name__)


def main(args=None):
    """Main entry point for opusfilter-slurm-submit command."""
    parser = argparse.ArgumentParser(
        prog='opusfilter-slurm-submit',
        description='Pre-submit OpusFilter workflow to SLURM without polling')
    parser.add_argument('config', metavar='CONFIG', help='YAML configuration file')
    parser.add_argument('--workdir', help='working directory for SLURM scripts and logs')
    parser.add_argument('--resume', '-r', help='skip completed steps', action='store_true')
    parser.add_argument('--dry-run', '-n',
                        help='show what would be done without submitting jobs',
                        action='store_true')
    parser.add_argument('--overwrite', help='overwrite existing output files', action='store_true')

    args = parser.parse_args(args)

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('mosestokenizer.tokenizer.MosesTokenizer').setLevel(logging.WARNING)

    config_path = os.path.abspath(args.config)
    configuration = yaml.load(open(config_path))
    configuration['_config_file'] = config_path

    workdir = args.workdir or os.path.expandvars(
        configuration.get('common', {}).get('slurm', {}).get(
            'workdir', '${HOME}/opusfilter-work'))
    workdir = os.path.abspath(os.path.expandvars(workdir))

    manifest_path = os.path.join(workdir, 'manifest.json')
    if os.path.exists(manifest_path) and not args.resume and not args.overwrite:
        logger.error(
            f"Manifest already exists at {manifest_path}. "
            "Use --overwrite to replace or --resume to continue.")
        return 1

    slurm_filter = SlurmOpusFilter(
        configuration,
        workdir=workdir,
        dry_run=args.dry_run)

    try:
        job_ids, graph, steps = slurm_filter.submit_all_steps(
            overwrite=args.overwrite,
            resume=args.resume)
    except Exception as e:
        logger.error(f"Failed to submit workflow: {e}")
        return 1

    if args.dry_run:
        logger.info("Dry run complete. No jobs submitted.")
        return 0

    logger.info(f"Manifest written to {manifest_path}")
    logger.info(f"Use 'opusfilter-slurm-status {workdir}' to monitor progress.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
