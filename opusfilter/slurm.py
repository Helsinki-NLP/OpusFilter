"""SLURM integration for OpusFilter workflows."""
import os
import subprocess
import time
import logging
from pathlib import Path

# Handle circular import by importing at runtime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from opusfilter.opusfilter import OpusFilter as OpusFilterMain
from opusfilter import ConfigurationError
from .util import count_lines
from .slurm_utils import (
    submit_job, get_job_status, cancel_job,
    build_dependency_graph, get_ready_steps,
    check_step_outputs, clean_failed_outputs
)

logger = logging.getLogger(__name__)


class SlurmOpusFilter:
    """Manages OpusFilter workflows on SLURM clusters."""
    
    def __init__(self, configuration, workdir=None, max_concurrent=None, 
                 email=None, dry_run=False):
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
    
    def run(self, overwrite=False, resume=False, monitor=False):
        """Run the workflow on SLURM."""
        logger.info(f"Starting SLURM workflow execution (dry_run={self.dry_run})")
        
        steps = self.configuration.get('steps', [])
        if not steps:
            logger.warning("No steps defined in configuration")
            return
        
        # Build dependency graph
        graph = build_dependency_graph(steps)
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
                completed_steps = [f"{i}_{steps[i]['type']}" 
                                for i in range(last_completed + 1)]
        
        # Main execution loop
        step_index = 0
        while completed_steps or step_index < len(steps):
            # Find ready steps
            ready = get_ready_steps(graph, completed_steps)
            if not ready:
                if running_jobs:
                    # Check status of running jobs
                    self._check_running_jobs(running_jobs)
                    time.sleep(10)
                    continue
                else:
                    # No ready steps and no running jobs - something's wrong
                    logger.error("Workflow deadlock detected!")
                    break
            
            # Submit as many as allowed up to max_concurrent
            to_submit = ready[:self.max_concurrent - len(running_jobs)]
            
            for step_name in to_submit:
                step_info = graph[step_name]
                step_index = step_info['index']
                step_config = step_info['step']
                
                # Check if outputs already exist
                if not overwrite and check_step_outputs(step_config, self.output_dir):
                    logger.info(f"Step {step_index} ({step_config['type']}) outputs exist, skipping")
                    completed_steps.append(step_name)
                    graph[step_name]['completed'] = True
                    continue
                
                # Get dependencies for this step
                deps = step_info['deps']
                dependency_id = None
                if deps:
                    # Use the first dependency (SLURM handles chain dependencies)
                    dep_names = [d for d in deps if d in self.job_ids]
                    if dep_names:
                        dependency_id = self.job_ids[dep_names[0]]
                
                # Submit job
                try:
                    job_id = self._submit_step(step_index, step_config, dependency_id)
                    self.job_ids[step_name] = job_id
                    running_jobs[step_name] = job_id
                    logger.info(f"Submitted step {step_index} ({step_config['type']}) as job {job_id}")
                except Exception as e:
                    logger.error(f"Failed to submit step {step_index}: {e}")
                    break
            
            step_index += len(to_submit)
        
        # Monitor final jobs
        if running_jobs:
            logger.info("Waiting for final jobs to complete...")
            self._wait_for_completion(list(running_jobs.values()))
        
        # Print summary
        self._print_summary(completed_steps)
        
        if monitor:
            self._monitor_jobs()
    
    def _submit_step(self, step_index, step_config, dependency_id=None):
        """Submit a single step as a SLURM job."""
        # Get SLURM resources for this step
        step_type = step_config['type']
        resources = self._get_step_resources(step_type)
        
        # Create batch script
        script_path = self._create_batch_script(step_index, step_config, resources, dependency_id)
        
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
    
    def _create_batch_script(self, step_index, step_config, resources, dependency_id=None):
        """Create a SLURM batch script for a step."""
        step_type = step_config['type']
        
        # Script path
        script_name = f"step_{step_index}_{step_type}.sbatch"
        script_path = os.path.join(self.workdir, "scripts", script_name)
        
        # Prepare template variables
        template_vars = {
            'job_name': f"opusfilter_{step_index}_{step_type}",
            'partition': resources.get('partition', 'cpu'),
            'account': self.slurm_config.get('account', ''),
            'time': resources.get('time', '02:00:00'),
            'mem': resources.get('mem', '4G'),
            'cpus': resources.get('cpus-per-task', 2),
            'log_dir': os.path.join(self.workdir, "logs"),
            'mail_type': self.slurm_config.get('mail_type', 'END,FAIL'),
            'mail_user': self.email or '',
            'step_index': step_index,
            'work_dir': self.workdir,
            'output_dir': self.output_dir,
            'module_loads': '',
            'command': '',
            'cleanup_command': '',
            'array_spec': '',
            'gres_spec': '',
            'dependency_spec': ''
        }
        
        # Add array specification
        if 'array_size' in resources:
            template_vars['array_spec'] = (
                f"#SBATCH --array=0-{resources['array_size']-1}%{self.max_concurrent}\n"
            )
        
        # Add GPU specification
        if 'gres' in resources:
            template_vars['gres_spec'] = f"#SBATCH --gres={resources['gres']}\n"
        
        # Add dependency specification
        if dependency_id:
            template_vars['dependency_spec'] = f"#SBATCH --dependency=afterok:{dependency_id}\n"
        
        # Add module loads
        modules = resources.get('modules', [])
        if modules:
            global_modules = self.slurm_config.get('modules', [])
            all_modules = global_modules + modules
            template_vars['module_loads'] = '\n'.join(f"module load {m}" for m in all_modules)
        
        # Generate command
        opusfilter_cmd = [
            'python', '-m', 'opusfilter.cli.main',
            os.path.abspath(self.configuration.get('_config_file', 'config.yaml')),
            '--single', str(step_index + 1)
        ]
        if overwrite and self.output_dir:
            opusfilter_cmd.append('--overwrite')
        
        template_vars['command'] = ' '.join(opusfilter_cmd)
        
        # Generate cleanup command
        template_vars['cleanup_command'] = f"""
        # Clean outputs on failure
        for output in {' '.join(step_config.get('parameters', {}).get('outputs', []))}; do
            if [ -f "${{OUTPUT_DIR}}/$output" ]; then
                rm "${{OUTPUT_DIR}}/$output"
            fi
        done
        """
        
        # Create script from template
        template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err
#SBATCH --mail-type={mail_type}
#SBATCH --mail-user={mail_user}
{array_spec}
{gres_spec}
{dependency_spec}

# Set exit on error
set -e

# Load modules
{module_loads}

# Environment setup
export OPUSFILTER_STEP={step_index}
export OUTPUT_DIR={output_dir}
export PYTHONUNBUFFERED=1

# Change to work directory
cd {work_dir}

# Create output directory if needed
mkdir -p {output_dir}

# Execute command
{command}

# Check exit code
if [ $? -eq 0 ]; then
    echo "Step completed successfully"
else
    echo "Step failed with exit code $?"
    {cleanup_command}
fi
"""
        
        with open(script_path, 'w') as f:
            f.write(template.format(**template_vars))
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        return script_path
    
    def _check_running_jobs(self, running_jobs):
        """Check status of running jobs."""
        to_remove = []
        for step_name, job_id in running_jobs.items():
            status = get_job_status(job_id)
            if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
                if status == 'COMPLETED':
                    logger.info(f"Step {step_name} completed successfully")
                    to_remove.append(step_name)
                else:
                    logger.error(f"Step {step_name} failed with status {status}")
                to_remove.append(step_name)
        
        for step_name in to_remove:
            del running_jobs[step_name]
    
    def _wait_for_completion(self, job_ids):
        """Wait for all jobs to complete."""
        while job_ids:
            for job_id in job_ids[:]:
                status = get_job_status(job_id)
                if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
                    job_ids.remove(job_id)
            if job_ids:
                time.sleep(30)
    
    def _find_last_completed_step(self, steps):
        """Find the last completed step based on output files."""
        for i in range(len(steps) - 1, -1, -1):
            if check_step_outputs(steps[i], self.output_dir):
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
    
    def _monitor_jobs(self):
        """Interactive job monitoring."""
        logger.info("Starting job monitoring (press Ctrl+C to stop)")
        try:
            while True:
                # Show job status
                subprocess.run(['squeue', '-u', self.email])
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")