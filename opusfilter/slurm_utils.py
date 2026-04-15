"""Utilities for SLURM job and dependency management."""
import copy
import json
import subprocess
import re
import logging
from pathlib import Path
from datetime import datetime

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
        result = subprocess.run(
            ['sacct', '-j', job_id, '-no', '--state=failed,completed,timedout,cancelled',
             '--format=State'],
            capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            state = result.stdout.strip().split('\n')[-1].strip()
            if state in ['COMPLETED', 'FAILED', 'TIMEDOUT', 'CANCELLED']:
                return state
        return 'UNKNOWN'
    status = result.stdout.strip('" \n')
    if not status:
        return 'UNKNOWN'
    return status


def is_job_completed(job_id):
    """Check if a job has completed (successfully or not)."""
    status = get_job_status(job_id)
    return status in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEDOUT']


def get_job_final_status(job_id):
    """Get final status of a job (completed or failed), or None if still running."""
    status = get_job_status(job_id)
    if status in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEDOUT']:
        return status
    return None


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


def write_manifest(path, config_file, steps, graph, job_ids, workdir):
    """Write workflow manifest to JSON.

    Args:
        path: Path to write manifest JSON file
        config_file: Path to original configuration file
        steps: List of expanded steps
        graph: Dependency graph dict
        job_ids: Dict mapping step_name -> SLURM job ID
        workdir: Working directory path
    """
    manifest = {
        "version": "1.0",
        "config": str(config_file),
        "submitted_at": datetime.now().isoformat(),
        "workdir": str(workdir),
        "jobs": {}
    }

    for step_name, step_info in graph.items():
        job_id = job_ids.get(step_name)
        if job_id:
            script_path = _get_script_path(workdir, step_info)
            manifest["jobs"][job_id] = {
                "step": step_name,
                "original_step": step_info.get("original_index", 0),
                "substep_index": step_info.get("substep_index"),
                "step_type": step_info["step"].get("type", "unknown"),
                "deps": step_info.get("deps", []),
                "script": script_path
            }

    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)


def read_manifest(path):
    """Read workflow manifest from JSON file.

    Args:
        path: Path to manifest JSON file or directory containing manifest.json

    Returns:
        Manifest dict with 'jobs', 'config', 'workdir', etc.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path) as f:
        return json.load(f)


def _get_script_path(workdir, step_info):
    """Generate expected script path for a step."""
    step_type = step_info["step"].get("type", "unknown")
    original_idx = step_info.get("original_index", 0) + 1
    substep_idx = step_info.get("substep_index")

    if substep_idx is not None:
        script_name = f"step_{original_idx}_{step_type}_{substep_idx + 1}.sbatch"
    else:
        script_name = f"step_{original_idx}_{step_type}.sbatch"

    return str(Path(workdir) / "scripts" / script_name)


def get_detailed_job_status(job_id):
    """Get detailed job status from SLURM.

    Args:
        job_id: SLURM job ID

    Returns:
        Dict with 'status', 'runtime', 'node', 'job_name'
    """
    result = {
        "status": "UNKNOWN",
        "runtime": None,
        "node": None,
        "job_name": None
    }

    # Try squeue first for running jobs
    proc = subprocess.run(
        ['squeue', '-j', job_id, '-h', '--format="%j|%T|%l|%N"'],
        capture_output=True, text=True)
    if proc.returncode == 0 and proc.stdout.strip():
        parts = proc.stdout.strip().strip('"').split('|')
        if len(parts) >= 4:
            result["job_name"] = parts[0]
            result["status"] = parts[1]
            result["runtime"] = parts[2]
            result["node"] = parts[3] if parts[3] != "N/A" else None
        elif len(parts) >= 2:
            result["status"] = parts[1]
        return result

    # Fallback to sacct for completed/failed jobs
    proc = subprocess.run(
        ['sacct', '-j', job_id, '-no', '--state=failed,completed,timedout,cancelled',
         '--format=JobName,State,Elapsed,NodeList'],
        capture_output=True, text=True)
    if proc.returncode == 0 and proc.stdout.strip():
        lines = proc.stdout.strip().split('\n')
        for line in reversed(lines):
            parts = line.strip().split('|')
            if len(parts) >= 4 and parts[0]:
                result["job_name"] = parts[0]
                result["status"] = parts[1]
                result["runtime"] = parts[2] if parts[2] != "00:00:00" else None
                result["node"] = parts[3] if parts[3] != "(" else None
                break
        if result["status"] == "UNKNOWN" and len(lines) > 0:
            parts = lines[-1].strip().split('|')
            if len(parts) >= 2:
                result["status"] = parts[1]

    return result


def get_manifest_status(manifest):
    """Get status for all jobs in a manifest.

    Args:
        manifest: Manifest dict from read_manifest()

    Returns:
        List of dicts with job_id, step, status, runtime, node
    """
    results = []
    for job_id, job_info in manifest.get("jobs", {}).items():
        details = get_detailed_job_status(job_id)
        results.append({
            "job_id": job_id,
            "step": job_info["step"],
            "step_type": job_info.get("step_type", "unknown"),
            "status": details["status"],
            "runtime": details["runtime"],
            "node": details["node"]
        })

    # Sort by original_step, then substep_index
    def sort_key(r):
        step_parts = r["step"].split("_")
        original = int(step_parts[0]) if step_parts[0].isdigit() else 0
        substep = int(step_parts[-1]) if step_parts[-1].isdigit() else 0
        return (original, substep)

    results.sort(key=sort_key)
    return results
