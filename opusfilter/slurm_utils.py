"""Utilities for SLURM job and dependency management."""
import subprocess
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def submit_job(script_path, dependency=None, array_size=None):
    """Submit a SLURM job and return job ID."""
    cmd = ['sbatch', script_path]

    if dependency:
        cmd.insert(1, f'--dependency=afterok:{dependency}')

    if array_size:
        cmd.insert(1, f'--array=0-{array_size-1}')

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr}")

    # Extract job ID from output
    match = re.search(r'Submitted batch job (\d+)', result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse job ID from: {result.stdout}")

    return match.group(1)


def get_job_status(job_id):
    """Get job status from SLURM."""
    result = subprocess.run(['squeue', '-j', job_id, '-h'],
                        capture_output=True, text=True)
    if result.returncode != 0:
        return 'UNKNOWN'

    for line in result.stdout.split('\n'):
        if line.startswith('JobState='):
            return line.split('=')[1]

    return 'UNKNOWN'


def cancel_job(job_id):
    """Cancel a SLURM job."""
    subprocess.run(['scancel', job_id])


def build_dependency_graph(steps):
    """Build dependency graph from step configurations."""
    graph = {}
    for i, step in enumerate(steps):
        step_name = f"{i}_{step['type']}"
        params = step.get('parameters', {})

        # Collect all possible outputs from different field names
        # FIXME: cli/diagram.py has get_outputs() for the same purpose
        outputs = []
        if 'output' in params:
            outputs.extend(params['output'])
        if 'outputs' in params:
            outputs.extend(params['outputs'])
        if 'src_output' in params:
            outputs.append(params['src_output'])
        if 'tgt_output' in params:
            outputs.append(params['tgt_output'])

        graph[step_name] = {
            'step': step,
            'index': i,
            'deps': [],
            'outputs': outputs
        }

    # Find dependencies
    for step_name, step_info in graph.items():
        inputs = step_info['step'].get('parameters', {}).get('inputs', [])
        if inputs:
            for input_file in inputs:
                for other_name, other_info in graph.items():
                    if input_file in other_info['outputs']:
                        step_info['deps'].append(other_name)

    return graph


def get_ready_steps(graph, completed_jobs):
    """Get steps whose dependencies are satisfied."""
    ready = []
    for step_name, step_info in graph.items():
        deps = step_info.get('deps', [])
        if all(dep in completed_jobs for dep in deps):
            # Only add steps that are not completed
            if not step_info.get('completed', False):
                ready.append(step_name)
    return ready


def check_step_outputs(step, output_dir):
    """Check if all outputs exist and are non-empty."""
    outputs = step.get('parameters', {}).get('outputs', [])
    if not outputs:
        return True
    for output in outputs:
        path = Path(output_dir) / output
        if not path.exists():
            return False
        if path.stat().st_size == 0:
            logger.warning(f"Output file {output} is empty")
            return False
    return True


def clean_failed_outputs(step, output_dir):
    """Remove outputs from failed step."""
    outputs = step.get('parameters', {}).get('outputs', [])
    for output in outputs:
        path = Path(output_dir) / output
        if path.exists():
            path.unlink()
