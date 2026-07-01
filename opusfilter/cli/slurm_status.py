"""Command-line interface for opusfilter-slurm-status command."""
import argparse
import logging
import sys
import json

from opusfilter.slurm_utils import read_manifest, get_manifest_status, cancel_job

logger = logging.getLogger(__name__)

DEFAULT_WATCH_INTERVAL = 30


def main(args=None):
    """Main entry point for opusfilter-slurm-status command."""
    parser = argparse.ArgumentParser(
        prog='opusfilter-slurm-status',
        description='Check status of SLURM workflow from manifest')
    parser.add_argument('path', metavar='PATH',
                        help='Manifest JSON file or workdir containing manifest.json')
    parser.add_argument('--json', action='store_true', help='Output status as JSONLines')
    parser.add_argument('--cancel', action='store_true',
                        help='Cancel remaining jobs if any have failed')
    parser.add_argument('--watch', '-w', action='store_true',
                        help='Watch mode: refresh status periodically')
    parser.add_argument('--watch-interval', type=int, default=DEFAULT_WATCH_INTERVAL,
                        help=f'Seconds between refresh in watch mode (default: {DEFAULT_WATCH_INTERVAL})')

    args = parser.parse_args(args)

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    try:
        manifest = read_manifest(args.path)
    except FileNotFoundError as e:
        logger.error(e)
        return 1

    try:
        return run_status(
            manifest,
            output_json=args.json,
            cancel_on_failure=args.cancel,
            watch=args.watch,
            watch_interval=args.watch_interval)
    except KeyboardInterrupt:
        return 0


def run_status(manifest, output_json=False, cancel_on_failure=False,
               watch=False, watch_interval=DEFAULT_WATCH_INTERVAL):
    """Run status check with optional watch mode."""
    while True:
        statuses = get_manifest_status(manifest)

        if output_json:
            for status in statuses:
                output = {
                    "job_id": status["job_id"],
                    "step": status["step"],
                    "status": status["status"],
                    "runtime": status["runtime"],
                    "node": status["node"],
                    "deps": status.get("deps", []),
                    "dep_jobs": status.get("dep_jobs", [])
                }
                print(json.dumps(output))
        else:
            print_status_table(statuses)

        failed_jobs = [s for s in statuses if s['status'] in ('FAILED', 'CANCELLED', 'TIMEDOUT')]
        completed_jobs = [s for s in statuses if s['status'] == 'COMPLETED']
        running_jobs = [s for s in statuses if s['status'] in ('RUNNING', 'PENDING', 'CONFIGURING')]
        total_jobs = len(statuses)

        print()
        summary = (
            f"{len(completed_jobs)}/{total_jobs} completed, "
            f"{len(running_jobs)} running, "
            f"{len(failed_jobs)} failed"
        )

        if failed_jobs:
            print(f"Status: FAILED - {summary}")
            if cancel_on_failure:
                return cancel_remaining_jobs(manifest, statuses)
            else:
                print()
                response = input("Failed jobs detected. Cancel remaining jobs? [y/N] ")
                if response.lower() == 'y':
                    return cancel_remaining_jobs(manifest, statuses)
        else:
            if len(completed_jobs) == total_jobs:
                print(f"Status: COMPLETE - {summary}")
                return 0
            print(f"Status: Running - {summary}")

        if not watch:
            return 0

        import time
        time.sleep(watch_interval)


def print_status_table(statuses):
    """Print status as a formatted table."""
    if not statuses:
        print("No jobs in manifest")
        return

    header = (
        f"{'JobID':<10} {'Step':<28} {'Status':<12} {'Runtime':<10} "
        f"{'Node':<12} {'Depends On':<22}"
    )
    separator = "-" * len(header)

    print(header)
    print(separator)

    for status in statuses:
        job_id = status['job_id']
        step = status['step']
        step_type = status.get('step_type', '')
        if step_type and step_type not in step:
            step_display = f"{step} ({step_type})"
        else:
            step_display = step

        status_str = status['status']
        runtime = status['runtime'] or '-'
        node = status['node'] or '-'
        dep_jobs = status.get('dep_jobs', [])
        if dep_jobs:
            step_deps = status.get('deps', [])
            if len(dep_jobs) > 1:
                deps_display = f"{step_deps[0]} ({dep_jobs[0]}) +{len(dep_jobs)-1} more"
            elif step_deps:
                deps_display = f"{step_deps[0]} ({dep_jobs[0]})"
            else:
                deps_display = dep_jobs[0]
        else:
            deps_display = '-'

        print(
            f"{job_id:<10} {step_display:<28} {status_str:<12} "
            f"{runtime:<10} {node:<12} {deps_display:<22}"
        )


def cancel_remaining_jobs(manifest, statuses):
    """Cancel all pending/running jobs."""
    pending = [s for s in statuses if s['status'] in ('RUNNING', 'PENDING', 'CONFIGURING')]

    if not pending:
        print("No pending jobs to cancel")
        return 0

    print(f"Cancelling {len(pending)} pending jobs...")
    for status in pending:
        print(f"  Cancelling job {status['job_id']} ({status['step']})")
        cancel_job(status['job_id'])

    print("Remaining jobs cancelled")
    return 0


if __name__ == '__main__':
    sys.exit(main())
