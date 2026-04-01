"""Utilities for SLURM job and dependency management."""
import copy
import subprocess
import re
import logging
from pathlib import Path

from opusfilter.util import get_inputs, get_outputs, expand_step_parameters

logger = logging.getLogger(__name__)


def submit_job(script_path, dependency=None, array_size=None):
    """Submit a SLURM job and return job ID.

    Args:
        script_path: Path to the SLURM batch script
        dependency: Single job ID (str) or collection of job IDs (list/set)
                    for afterok dependency. Multiple deps are colon-separated.
        array_size: If set, create an array job with this many elements
    """
    cmd = ['sbatch', script_path]

    if dependency:
        if isinstance(dependency, (list, set, tuple)):
            dep_str = ':'.join(str(d) for d in dependency if d)
            if dep_str:
                cmd.insert(1, f'--dependency=afterok:{dep_str}')
        else:
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
    result = subprocess.run(
        ['squeue', '-j', job_id, '-h', '--format="%T"', '--states=all'],
        capture_output=True, text=True)
    if result.returncode != 0:
        # Already completed and no longer seen by squeue?
        result = subprocess.run(
            ['sacct', '-j', job_id, '--noheader', '--allocations', '--format=State'],
            capture_output=True, text=True)
        if result.returncode != 0:
            return 'UNKNOWN'
    return result.stdout.strip('" \n')


def cancel_job(job_id):
    """Cancel a SLURM job."""
    subprocess.run(['scancel', job_id])


def build_dependency_graph(steps):
    """Build dependency graph from step configurations.

    For expanded steps (with _expanded_parameters), uses resolved parameters
    to determine inputs/outputs. For zipped dependencies (same original step
    with variables), matches substeps by _substep_index.

    Supports explicit dependencies via 'depends_on' field in step config,
    which should contain a list of output filenames that this step depends on.
    """
    graph = {}
    for i, step in enumerate(steps):
        step_name = _get_step_name(step, i)

        # Use expanded parameters if available
        if '_expanded_parameters' in step:
            expanded_params = step['_expanded_parameters']
            outputs = get_outputs({'type': step.get('type'), 'parameters': expanded_params})
        else:
            outputs = get_outputs(step)

        graph[step_name] = {
            'step': step,
            'index': i,
            'original_index': step.get('_original_index', i),
            'substep_index': step.get('_substep_index'),
            'deps': [],
            'outputs': outputs
        }

    # Find dependencies from inputs/outputs matching
    for step_name, step_info in graph.items():
        # Use expanded parameters if available
        if '_expanded_parameters' in step_info['step']:
            expanded_params = step_info['step']['_expanded_parameters']
            step = copy.copy(step_info['step'])
            step['parameters'] = expanded_params
        else:
            step = step_info['step']
        inputs = get_inputs(step)

        if inputs:
            for input_file in inputs:
                for other_name, other_info in graph.items():
                    if input_file in other_info['outputs'] and other_name not in step_info['deps']:
                        orig_idx = step_info.get('original_index')
                        other_orig_idx = other_info.get('original_index')
                        sub_idx = step_info.get('substep_index')
                        other_sub_idx = other_info.get('substep_index')

                        # Zipped dependency: same original step with variables
                        if orig_idx == other_orig_idx and sub_idx is not None and other_sub_idx is not None:
                            # Same original step - match by substep_index only
                            if sub_idx == other_sub_idx:
                                step_info['deps'].append(other_name)
                        elif orig_idx != other_orig_idx:
                            # Different original steps - regular dependency
                            step_info['deps'].append(other_name)

    return graph


def _get_step_name(step, index):
    """Generate a unique name for a step."""
    step_type = step.get('type', 'unknown')
    substep_idx = step.get('_substep_index')
    original_idx = step.get('_original_index', index)

    if substep_idx is not None:
        return f"{original_idx}_{step_type}_{substep_idx}"
    return f"{original_idx}_{step_type}"


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


def check_step_outputs(step, output_dir, constants=None):
    """Check if all outputs exist and are non-empty.

    For expanded steps, uses _expanded_parameters if available.
    """
    # Use expanded parameters if available
    if '_expanded_parameters' in step:
        expanded_params = step['_expanded_parameters']
    else:
        constants = constants or {}
        namespace = copy.copy(constants)
        namespace.update(step.get('constants', {}))
        expanded_params = expand_step_parameters(step.get('parameters', {}), namespace)

    outputs = get_outputs({'type': step.get('type'), 'parameters': expanded_params})
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


def clean_failed_outputs(step, output_dir, constants=None):
    """Remove outputs from failed step.

    For expanded steps, uses _expanded_parameters if available.
    """
    # Use expanded parameters if available
    if '_expanded_parameters' in step:
        expanded_params = step['_expanded_parameters']
    else:
        constants = constants or {}
        namespace = copy.copy(constants)
        namespace.update(step.get('constants', {}))
        expanded_params = expand_step_parameters(step.get('parameters', {}), namespace)

    outputs = get_outputs({'type': step.get('type'), 'parameters': expanded_params})
    for output in outputs:
        path = Path(output_dir) / output
        if path.exists():
            path.unlink()
