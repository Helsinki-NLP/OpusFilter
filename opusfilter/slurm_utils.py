"""Utilities for SLURM job and dependency management."""
import copy
import subprocess
import re
import logging
from pathlib import Path

from opusfilter.util import get_inputs, get_outputs
from opusfilter.util import VarStr, Var

logger = logging.getLogger(__name__)


def expand_step_parameters(step, constants):
    """Expand Var and VarStr objects in step parameters using constants and variables."""
    variables = step.get('variables', {})
    namespace = copy.copy(constants)
    namespace.update(step.get('constants', {}))

    if variables:
        num_choices = len(next(iter(variables.values()), []))
        if num_choices > 0:
            idx = 0
            for key, values in variables.items():
                namespace[key] = values[idx]

    def expand_obj(obj):
        if isinstance(obj, list):
            return [expand_obj(x) for x in obj]
        if isinstance(obj, dict):
            return {expand_obj(k): expand_obj(v) for k, v in obj.items()}
        if isinstance(obj, VarStr):
            try:
                return obj.value.format(**namespace)
            except (KeyError, IndexError):
                return obj.value
        if isinstance(obj, Var):
            return namespace.get(obj.value, obj.value)
        return obj

    params = step.get('parameters', {})
    return expand_obj(params)


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
    """Build dependency graph from step configurations."""
    graph = {}
    for i, step in enumerate(steps):
        step_name = f"{i}_{step['type']}"
        outputs = get_outputs(step)

        graph[step_name] = {
            'step': step,
            'index': i,
            'deps': [],
            'outputs': outputs
        }

    # Find dependencies
    for step_name, step_info in graph.items():
        inputs = get_inputs(step_info['step'])
        if inputs:
            for input_file in inputs:
                for other_name, other_info in graph.items():
                    if input_file in other_info['outputs'] and not other_name in step_info['deps']:
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


def check_step_outputs(step, output_dir, constants=None):
    """Check if all outputs exist and are non-empty."""
    constants = constants or {}
    expanded_params = expand_step_parameters(step, constants)
    outputs = get_outputs({'parameters': expanded_params})
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
    """Remove outputs from failed step."""
    constants = constants or {}
    expanded_params = expand_step_parameters(step, constants)
    outputs = get_outputs({'parameters': expanded_params})
    for output in outputs:
        path = Path(output_dir) / output
        if path.exists():
            path.unlink()
