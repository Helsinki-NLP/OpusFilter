"""SLURM integration for OpusFilter workflows."""
import os
import time
import logging
from pathlib import Path

from .opusfilter import OpusFilter
from .util import convert_vars_to_strings, expand_steps_with_variables
from .slurm_utils import (
    submit_job, get_job_status, is_job_completed,
    build_dependency_graph, get_ready_steps,
    check_step_outputs, _get_step_name
)
from .validate import validate_configuration


TEMPLATE_DIR = Path(__file__).parent.parent / 'templates' / 'slurm'
STEP_TEMPLATE_PATH = TEMPLATE_DIR / 'step_template.sbatch'


logger = logging.getLogger(__name__)


class SlurmOpusFilter:
    """Manages OpusFilter workflows on SLURM clusters."""

    def __init__(self, configuration, workdir=None, max_concurrent=None,
                 email=None, dry_run=False):
        validate_configuration(configuration)
        self.configuration = configuration
        self.output_dir = configuration.get('common', {}).get('output_directory', '.')
        self.workdir = workdir or os.path.expandvars(
            configuration.get('common', {}).get('slurm', {}).get('workdir', '${HOME}/opusfilter-work'))
        self.max_concurrent = max_concurrent or 4
        self.email = email or configuration.get('common', {}).get('slurm', {}).get('mail_user')
        self.dry_run = dry_run
        self.slurm_config = configuration.get('common', {}).get('slurm', {})
        self.job_ids = {}  # Track submitted jobs

        # Ensure workdir exists
        os.makedirs(self.workdir, exist_ok=True)
        os.makedirs(f"{self.workdir}/logs", exist_ok=True)
        os.makedirs(f"{self.workdir}/scripts", exist_ok=True)

        # Load OpusFilter
        self.opusfilter = OpusFilter(configuration)

    def run(self, overwrite=False, resume=False):
        """Run the workflow on SLURM."""
        logger.info(f"Starting SLURM workflow execution (dry_run={self.dry_run})")

        original_steps = self.configuration.get('steps', [])
        if not original_steps:
            logger.warning("No steps defined in configuration")
            return

        # Expand steps with variables into individual substeps
        common_constants = self.configuration.get('common', {}).get('constants', {})
        steps = expand_steps_with_variables(original_steps, common_constants)
        logger.info(f"Expanded {len(original_steps)} steps into {len(steps)} substeps")

        # Build dependency graph
        graph = build_dependency_graph(steps)
        for key, value in graph.items():
            logger.debug("Dependencies found for %s: %s", key, value['deps'])
        completed_steps = []
        running_jobs = {}

        # Resume from last completed step if requested
        if resume:
            last_completed = self._find_last_completed_step(steps)
            if last_completed is not None:
                logger.info(f"Resuming from step {last_completed + 1}")
                steps = steps[last_completed + 1:]
                # Update graph
                graph = build_dependency_graph(steps)
                completed_steps = [
                    _get_step_name(original_steps[i], i)
                    for i in range(last_completed + 1)
                ]

        # Main execution loop
        failed_steps = []
        while True:
            # Find ready steps
            ready = get_ready_steps(graph, completed_steps)
            # Filter out steps that are already running
            ready = [s for s in ready if s not in running_jobs]

            if ready:
                logger.info(f"Loop: ready={ready}, running={list(running_jobs.keys())}, completed={completed_steps}")

                # Check if we can submit more (respect max_concurrent)
                slots_available = self.max_concurrent - len(running_jobs)
                if slots_available <= 0:
                    # At capacity - check status and wait
                    if running_jobs:
                        completed, failed = self._check_running_jobs(running_jobs, graph)
                        completed_steps.extend(completed)
                        failed_steps.extend(failed)
                        if failed_steps:
                            logger.error(f"Workflow failed: {len(failed_steps)} step(s) failed")
                            return False
                        if completed:
                            continue
                    logger.info(f"At max_concurrent ({self.max_concurrent}), waiting for jobs to complete...")
                    time.sleep(10)
                    continue

                # Submit as many as allowed up to max_concurrent
                to_submit = ready[:slots_available]

                # First pass: collect dependency job IDs and handle completed deps
                step_deps = {}
                batch_job_ids = {}
                newly_completed = []  # Track dependencies that completed during this loop
                for step_name in to_submit:
                    step_info = graph[step_name]
                    deps = step_info['deps']

                    dep_ids = set()
                    for dep in deps:
                        if dep in self.job_ids:
                            dep_id = self.job_ids[dep]
                            if is_job_completed(dep_id):
                                logger.info(
                                    f"Step {step_name}: dependency {dep} "
                                    f"(job {dep_id}) completed, not using as dependency")
                                newly_completed.append(dep)
                                continue
                            dep_ids.add(dep_id)
                        if dep in batch_job_ids:
                            dep_ids.add(batch_job_ids[dep])

                    step_deps[step_name] = dep_ids

                # Add newly completed dependencies to completed_steps if not already there
                for dep in newly_completed:
                    if dep not in completed_steps:
                        completed_steps.append(dep)

                # Second pass: submit all jobs with their valid dependency lists
                for step_name in to_submit:
                    step_info = graph[step_name]
                    step_index = step_info['index']
                    step_config = step_info['step']

                    # Use original step index for CLI --single option (1-based)
                    original_step_index = step_info.get('original_index', step_index) + 1

                    # Check if outputs already exist
                    constants = self.configuration.get('common', {}).get('constants', {})
                    if not overwrite and check_step_outputs(step_config, self.output_dir, constants):
                        logger.info(f"Step {original_step_index} ({step_config['type']}) outputs exist, skipping")
                        completed_steps.append(step_name)
                        graph[step_name]['completed'] = True
                        continue

                    # Get valid dependency job IDs for this step
                    dependency_ids = step_deps[step_name]
                    if dependency_ids:
                        logger.info(f"Step {step_name} has dependencies: {step_info['deps']} -> job IDs: {dependency_ids}")
                    elif step_info['deps']:
                        logger.warning(f"Step {step_name} has deps {step_info['deps']} but none found in job_ids")

                    # Submit job
                    try:
                        job_id = self._submit_step(original_step_index, step_config, dependency_ids, overwrite)
                        self.job_ids[step_name] = job_id
                        batch_job_ids[step_name] = job_id
                        running_jobs[step_name] = job_id
                        logger.info(
                            f"Submitted step {original_step_index} ({step_config['type']}, "
                            f"substep {step_info.get('substep_index')}) as job {job_id}")
                    except Exception as e:
                        logger.error(f"Failed to submit step {original_step_index}: {e}")
                        break

                # After submitting, loop back immediately to check for newly ready steps
                continue

            # No ready steps
            if running_jobs:
                # Check status of running jobs
                logger.info(f"Waiting: running={list(running_jobs.keys())}, completed={completed_steps}")
                completed, failed = self._check_running_jobs(running_jobs, graph)
                completed_steps.extend(completed)
                failed_steps.extend(failed)

                # Stop if any step has failed
                if failed_steps:
                    logger.error(f"Workflow failed: {len(failed_steps)} step(s) failed")
                    return False

                # If we have new completed jobs, loop back to check for ready steps
                if completed:
                    continue

                # No new completions, wait before checking again
                logger.info("Waiting for jobs to complete...")
                time.sleep(10)
                continue

            # Check if all steps are completed
            all_completed = all(step_info.get('completed', False) for step_info in graph.values())
            if all_completed:
                break
            # No ready steps and no running jobs - something's wrong
            logger.error("Workflow deadlock detected!")
            break

        # Monitor final jobs
        if running_jobs:
            logger.info("Waiting for final jobs to complete...")
            success = self._wait_for_completion(running_jobs, graph, completed_steps)
            if not success:
                logger.error("Workflow failed: some jobs failed")
                return False

        # Check if any steps failed
        if any(step_info.get('failed', False) for step_info in graph.values()):
            logger.error("Workflow failed: some steps failed")
            return False

        # Print summary
        self._print_summary(completed_steps)

    def _submit_step(self, step_index, step_config, dependency_id=None, overwrite=False):
        """Submit a single step as a SLURM job."""
        # Get SLURM resources for this step
        step_type = step_config['type']
        resources = self._get_step_resources(step_type)

        # Create batch script
        script_path = self._create_batch_script(step_index, step_config, resources, dependency_id, overwrite)

        if self.dry_run:
            logger.info(f"[DRY RUN] Would submit: {script_path}")
            return f"dryrun_{step_index}"

        # Determine if this step should use array jobs
        array_size = None
        if 'array_size' in resources:
            array_size = resources['array_size']

        # Submit to SLURM
        return submit_job(script_path, dependency_id, array_size)

    def _get_step_resources(self, step_type):
        """Get SLURM resources for a step type."""
        # Start with global defaults
        resources = self.slurm_config.get('default', {}).copy()

        # Merge step-specific resources
        step_resources = self.slurm_config.get('resources', {}).get(step_type, {})
        resources.update(step_resources)

        return resources

    def _create_batch_script(self, step_index, step_config, resources, dependency_id=None, overwrite=False):
        """Create a SLURM batch script for a step."""
        step_type = step_config['type']
        substep_idx = step_config.get('_substep_index')

        if substep_idx is not None:
            script_name = f"step_{step_index + 1}_{step_type}_{substep_idx + 1}.sbatch"
        else:
            script_name = f"step_{step_index + 1}_{step_type}.sbatch"
        script_path = os.path.join(self.workdir, "scripts", script_name)

        partition = resources.get('partition') or self.slurm_config.get('partition') or 'cpu'
        job_name_suffix = f"_{substep_idx + 1}" if substep_idx is not None else ""
        template_vars = {
            'job_name': f"opusfilter_{step_index + 1}_{step_type}{job_name_suffix}",
            'partition': partition,
            'account': self.slurm_config.get('account', ''),
            'time': resources.get('time', '02:00:00'),
            'mem': resources.get('mem', '4G'),
            'cpus': resources.get('cpus-per-task', 2),
            'log_dir': os.path.join(self.workdir, "logs"),
            'mail_type': self.slurm_config.get('mail_type', 'END,FAIL'),
            'mail_user': self.email or '',
            'output_dir': self.output_dir,
            'module_loads': '',
            'command': '',
            'cleanup_command': '',
            'array_spec': '',
            'gres_spec': '',
            'dependency_spec': ''
        }

        if 'array_size' in resources:
            template_vars['array_spec'] = (
                f"#SBATCH --array=0-{resources['array_size']-1}%{self.max_concurrent}\n"
            )

        if 'gres' in resources:
            template_vars['gres_spec'] = f"#SBATCH --gres={resources['gres']}\n"

        if dependency_id:
            template_vars['dependency_spec'] = f"#SBATCH --dependency=afterok:{dependency_id}\n"

        modules = resources.get('modules', [])
        if modules:
            global_modules = self.slurm_config.get('modules', [])
            all_modules = global_modules + modules
            template_vars['module_loads'] = '\n'.join(f"module load {m}" for m in all_modules)

        # Generate command
        opusfilter_cmd = [
            'python', '-m', 'opusfilter.cli.main',
            os.path.abspath(self.configuration.get('_config_file', 'config.yaml')),
            '--single', str(step_index)
        ]
        if substep_idx is not None:
            opusfilter_cmd.extend(['--substep', str(substep_idx + 1)])  # 1-based
        if overwrite and self.output_dir:
            opusfilter_cmd.append('--overwrite')

        template_vars['command'] = ' '.join(opusfilter_cmd)

        # Generate cleanup command
        step_params = step_config.get('parameters', {})
        step_outputs = convert_vars_to_strings(step_params.get('outputs', []))
        if step_outputs:
            template_vars['cleanup_command'] = f"""
for output in {' '.join(step_outputs)}; do
    if [ -f "${{OUTPUT_DIR}}/$output" ]; then
        rm "${{OUTPUT_DIR}}/$output"
    fi
done"""

        # Load template from file
        template = STEP_TEMPLATE_PATH.read_text()

        with open(script_path, 'w') as f:
            f.write(template.format(**template_vars))

        os.chmod(script_path, 0o755)

        return script_path

    def _check_running_jobs(self, running_jobs, graph):
        """Check status of running jobs. Returns list of completed step names."""
        logger.info(f"Checking status of {len(running_jobs)} running jobs: {running_jobs}")
        to_remove = []
        failed_steps = []
        for step_name, job_id in running_jobs.items():
            if job_id.startswith('dryrun_'):
                logger.info(f"Step {step_name} completed successfully (dry run)")
                to_remove.append(step_name)
                graph[step_name]['completed'] = True
                continue
            status = get_job_status(job_id)
            logger.info(f"Job {job_id} ({step_name}) status: {status}")
            if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
                if status == 'COMPLETED':
                    logger.info(f"Step {step_name} completed successfully")
                    graph[step_name]['completed'] = True
                    to_remove.append(step_name)
                else:
                    logger.error(f"Step {step_name} failed with status {status}")
                    graph[step_name]['failed'] = True
                    to_remove.append(step_name)
                    failed_steps.append(step_name)

        completed = []
        for step_name in to_remove:
            del running_jobs[step_name]
            completed.append(step_name)
        return completed, failed_steps

    def _wait_for_completion(self, running_jobs, graph, completed_steps):
        """Wait for all jobs to complete. Returns True if all succeeded, False if any failed."""
        job_ids = list(running_jobs.values())
        failed_steps = []
        while job_ids:
            for job_id in job_ids[:]:
                if job_id.startswith('dryrun_'):
                    for step_name, jid in running_jobs.items():
                        if jid == job_id:
                            graph[step_name]['completed'] = True
                            completed_steps.append(step_name)
                    job_ids.remove(job_id)
                    continue
                status = get_job_status(job_id)
                if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
                    for step_name, jid in running_jobs.items():
                        if jid == job_id:
                            if status == 'COMPLETED':
                                graph[step_name]['completed'] = True
                                completed_steps.append(step_name)
                            else:
                                logger.error(f"Step {step_name} failed with status {status}")
                                graph[step_name]['failed'] = True
                                failed_steps.append(step_name)
                    job_ids.remove(job_id)
            if job_ids:
                time.sleep(30)
        return len(failed_steps) == 0

    def _find_last_completed_step(self, steps):
        """Find the last completed step based on output files."""
        constants = self.configuration.get('common', {}).get('constants', {})
        for i in range(len(steps) - 1, -1, -1):
            if check_step_outputs(steps[i], self.output_dir, constants):
                return i
        return None

    def _print_summary(self, completed_steps):
        """Print execution summary."""
        logger.info("=" * 50)
        logger.info("WORKFLOW EXECUTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Completed steps: {len(completed_steps)}")
        for step_name in completed_steps:
            logger.info(f"  ✓ {step_name}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 50)

    def submit_all_steps(self, overwrite=False, resume=False):
        """Submit all steps without polling.

        Returns:
            Tuple of (job_ids dict, graph dict, steps list) for manifest creation.
            job_ids maps step_name -> SLURM job ID
        """
        original_steps = self.configuration.get('steps', [])
        if not original_steps:
            logger.warning("No steps defined in configuration")
            return {}, {}, []

        common_constants = self.configuration.get('common', {}).get('constants', {})
        steps = expand_steps_with_variables(original_steps, common_constants)
        logger.info(f"Expanded {len(original_steps)} steps into {len(steps)} substeps")

        graph = build_dependency_graph(steps)
        for key, value in graph.items():
            logger.debug("Dependencies found for %s: %s", key, value['deps'])

        job_ids = {}
        completed_steps = []

        if resume:
            last_completed = self._find_last_completed_step(steps)
            if last_completed is not None:
                logger.info(f"Resuming: skipping {last_completed + 1} completed steps")
                completed_steps = [
                    _get_step_name(original_steps[i], i)
                    for i in range(last_completed + 1)
                ]
                completed_step_names = set(completed_steps)

                new_graph = {}
                for step_name, info in graph.items():
                    filtered_deps = [d for d in info['deps'] if d not in completed_step_names]
                    new_graph[step_name] = {**info, 'deps': filtered_deps}
                graph = new_graph

                for i, step in enumerate(steps[:last_completed + 1]):
                    step_name = _get_step_name(
                        original_steps[step.get('_original_index', i)],
                        step.get('_original_index', i))
                    job_ids[step_name] = "completed"

        while True:
            ready = get_ready_steps(graph, completed_steps)
            if not ready:
                break

            for step_name in ready:
                step_info = graph[step_name]
                step_index = step_info['index']
                step_config = step_info['step']
                original_step_index = step_info.get('original_index', step_index) + 1

                constants = self.configuration.get('common', {}).get('constants', {})
                if not overwrite and check_step_outputs(step_config, self.output_dir, constants):
                    logger.info(f"Step {original_step_index} ({step_config['type']}) outputs exist, skipping")
                    completed_steps.append(step_name)
                    graph[step_name]['completed'] = True
                    continue

                dep_ids = set()
                for dep in step_info['deps']:
                    if dep in job_ids:
                        dep_id = job_ids[dep]
                        if not is_job_completed(dep_id):
                            dep_ids.add(dep_id)

                try:
                    job_id = self._submit_step(original_step_index, step_config, dep_ids, overwrite)
                    job_ids[step_name] = job_id
                    completed_steps.append(step_name)
                    graph[step_name]['completed'] = True
                    logger.info(
                        f"Submitted step {original_step_index} ({step_config['type']}) "
                        f"as job {job_id}"
                    )
                except Exception as e:
                    logger.error(f"Failed to submit step {original_step_index}: {e}")
                    raise

        logger.info(f"Submitted {len(job_ids)} jobs total")
        return job_ids, graph, steps
