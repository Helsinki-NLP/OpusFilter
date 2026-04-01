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
{array_spec}{gres_spec}{dependency_spec}

set -e

{module_loads}

export OUTPUT_DIR={output_dir}
export PYTHONUNBUFFERED=1

mkdir -p {output_dir}

{command}

if [ $? -eq 0 ]; then
    echo "Step completed successfully"
else
    echo "Step failed with exit code $?"
    {cleanup_command}
fi
