#!/bin/bash
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
    # Clean outputs on failure if not overwrite
    {cleanup_command}
fi