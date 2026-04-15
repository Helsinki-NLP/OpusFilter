# OpusFilter SLURM Integration

OpusFilter supports running workflows on SLURM clusters with resource-optimized job scheduling.

## Overview

The SLURM integration provides three commands:

- `opusfilter-slurm-submit`: Pre-submit all jobs and exit immediately
- `opusfilter-slurm-run`: Run with polling and monitoring
- `opusfilter-slurm-status`: Check workflow status from a manifest

## Installation

The SLURM integration is part of the standard OpusFilter installation. No additional dependencies are required.

## Quick Start

1. Create a configuration with SLURM settings:

```yaml
common:
  output_directory: /scratch/myproject/output
  slurm:
    account: myaccount
    partition: cpu
    mail_user: user@institution.edu
    resources:
      filter:
        time: 04:00:00
        mem: 8G
        array_size: 8
      score:
        partition: gpu
        time: 08:00:00
        mem: 32G
        gres: gpu:1
        n_jobs: 4

steps:
  - type: opus_read
    parameters:
      # ...
  - type: filter
    parameters:
      # ...
```

2. Choose a mode:

```bash
# Option 1: Pre-submit and monitor separately (recommended for long workflows)
opusfilter-slurm-submit config.yaml --workdir /scratch/myproject-work
opusfilter-slurm-status /scratch/myproject-work --watch

# Option 2: Run with built-in monitoring
opusfilter-slurm-run config.yaml \
    --resume \
    --max-concurrent 10 \
    --workdir /scratch/myproject-work
```

## Commands

### opusfilter-slurm-submit

Pre-submit all SLURM jobs and exit immediately. Jobs run via SLURM dependencies without a persistent process.

```bash
opusfilter-slurm-submit CONFIG [--workdir DIR] [--resume] [--dry-run] [--overwrite]
```

Options:
- `--workdir DIR`: Working directory for scripts and logs
- `--resume`: Skip completed steps based on output files
- `--dry-run`: Show what would be submitted without submitting
- `--overwrite`: Overwrite existing manifest

After submission, a `manifest.json` file is created in the workdir. Use `opusfilter-slurm-status` to monitor progress.

### opusfilter-slurm-run

Run the workflow with polling and monitoring. This keeps a persistent process that monitors job completion and submits new jobs as dependencies are satisfied.

```bash
opusfilter-slurm-run CONFIG [--workdir DIR] [--resume] [--dry-run] [--overwrite]
                           [--max-concurrent N] [--email ADDRESS]
```

Options:
- `--workdir DIR`: Working directory for scripts and logs
- `--resume`: Skip completed steps based on output files
- `--dry-run`: Show what would be done without submitting
- `--overwrite`: Overwrite existing output files
- `--max-concurrent N`: Maximum concurrent jobs (default: 4)
- `--email ADDRESS`: Override email for notifications

### opusfilter-slurm-status

Check status of a pre-submitted workflow.

```bash
opusfilter-slurm-status PATH [--json] [--cancel] [--watch] [--watch-interval SECONDS]
```

Options:
- `--json`: Output status as JSONLines
- `--cancel`: Cancel remaining jobs if any have failed
- `--watch`: Watch mode: refresh status periodically
- `--watch-interval SECONDS`: Seconds between refresh (default: 30)

Example output:

```
JobID      Step                    Status       Runtime    Node
--------   ---------------------   ---------    --------   ------
123456     1_opus_read            COMPLETED    00:02:30   -
123457     2_filter_1             RUNNING      00:15:42   node07
123458     2_filter_2             RUNNING      00:01:23   node12
123459     3_score                PENDING      -          -

3/4 completed, 2 running, 0 failed
Status: Running - 3/4 completed, 2 running, 0 failed
```

## Configuration

### SLURM Section

Add a `slurm` section under `common` in your YAML configuration:

#### Global Settings

- `account`: SLURM account name
- `partition`: Default partition
- `qos`: Quality of Service
- `mail_type`: Email notification types
- `mail_user`: Email for notifications
- `workdir`: Directory for SLURM scripts and logs
- `modules`: Modules to load for all jobs
- `default`: Default resource specifications

#### Per-Step Resources

Under `slurm.resources`, specify resources per step type:

```yaml
resources:
  filter:
    time: 04:00:00    # Wall time limit
    mem: 8G              # Memory requirement
    cpus-per-task: 4      # CPU cores
    partition: gpu           # Override default partition
    gres: gpu:1            # GPU resources
    array_size: 8          # For parallel steps
    modules:               # Additional modules
      - cuda/11.8
```

## Features

### Dependency Management

- Automatic detection based on input/output file matching
- Support for complex workflows with branch dependencies
- No manual job ID tracking required

### Concurrent Execution

- Independent steps run simultaneously (up to `--max-concurrent` with `opusfilter-slurm-run`)
- Efficient resource utilization
- Reduced queue wait times

### Resource Optimization

- Per-step resource allocation
- Array job support for parallelizable steps
- GPU allocation for GPU-intensive tasks

### Monitoring and Resumption

- `--resume`: Skip completed steps, continue from failures
- Automatic cleanup of failed outputs

## Best Practices

1. **For Long Workflows**: Use `opusfilter-slurm-submit` + `opusfilter-slurm-status`
   - No persistent process needed
   - Run status checker in tmux or as a separate job
   - Can disconnect from login node while jobs run

2. **Estimate Resources**
   - Start with conservative time/memory limits
   - Check actual usage with `seff` after completion
   - Adjust based on historical data

3. **Organize Workflows**
   - Place I/O-intensive steps early (opus_read, concatenate)
   - Group similar resource requirements
   - Avoid unnecessary dependencies

4. **Use Arrays**
   - Enable array_size for filter/score steps
   - Parallelizes within step, not just between steps

## Example Workflow

```yaml
# Complete example with all features
common:
  output_directory: corpus/processed
  slurm:
    account: nlp_project
    partition: gpu
    mail_type: END,FAIL,TIME_LIMIT_90
    mail_user: researcher@university.edu
    workdir: ${SLURM_TMP}/opusfilter
    modules:
      - python/3.10
      - cuda/11.8
    default:
      time: 02:00:00
      mem: 4G
      cpus-per-task: 2
    resources:
      opus_read:
        time: 01:00:00
        mem: 2G
      train_ngram:
        time: 24:00:00
        mem: 32G
        cpus-per-task: 16
      filter:
        time: 12:00:00
        mem: 16G
        array_size: 16
      score:
        time: 06:00:00
        mem: 64G
        partition: gpu
        gres: gpu:2
        n_jobs: 8

steps:
  - type: opus_read
    parameters:
      corpus_name: OpenSubtitles
      source_language: en
      target_language: es
      src_output: raw.en.gz
      tgt_output: raw.es.gz

  - type: train_ngram
    parameters:
      data: raw.en.gz
      parameters:
        norder: 5
      model: en.arpa.gz

  - type: filter
    parameters:
      inputs: [raw.en.gz, raw.es.gz]
      outputs: [filtered.en.gz, filtered.es.gz]
      filters: &common_filters
        - LengthFilter:
            unit: word
            min_length: 1
            max_length: 100
        - LanguageIDFilter:
            languages: [en, es]

  - type: score
    parameters:
      inputs: [filtered.en.gz, filtered.es.gz]
      output: scores.jsonl.gz
      filters: *common_filters
```

## Troubleshooting

### Jobs Not Submitting
- Check account name and partition availability
- Verify module names
- Ensure workdir is writable
- Check with `--dry-run` first

### Jobs Failing
- Check logs in `${workdir}/logs/`
- Verify input/output paths
- Use `scontrol show job <id>` for details
- Ensure requested resources are available

### Performance Issues
- Reduce `array_size` if jobs are too small
- Increase memory if OOM errors occur
- Use appropriate partition (cpu/gpu)

### Status Check Shows Failures
If `opusfilter-slurm-status` shows failed jobs:

```bash
# View failed job details
opusfilter-slurm-status /scratch/work --json | grep FAILED

# Cancel remaining jobs
opusfilter-slurm-status /scratch/work --cancel
```

## Integration with Other Tools

The SLURM integration outputs standard OpusFilter files that can be used with:
- `opusfilter-diagram`: Visualize workflow
- `opusfilter-scores`: Analyze job scores
- Custom monitoring scripts

## Advanced Usage

### Explicit Dependencies (depends_on)

Some steps have implicit file dependencies that are not tracked through standard input/output fields. For example, `LMClassifierFilter` loads model files specified in `lm_params.*.filename`, but these are not automatically detected as dependencies.

Use the `depends_on` field to explicitly declare these dependencies:

```yaml
steps:
  # Step that produces model files
  - type: train_ngram
    parameters:
      data: data.txt.gz
      model: model.arpa.gz

  # Step that uses the model (implicit dependency via lm_params)
  - type: filter
    parameters:
      inputs: [data.txt.gz]
      outputs: [filtered.txt.gz]
      filters:
        - LMClassifierFilter:
            lm_params:
              en: {filename: model.arpa.gz}
    depends_on:
      - model.arpa.gz
```

The `depends_on` field supports:
- Single file (string): `depends_on: model.arpa.gz`
- Multiple files (list): `depends_on: [file1.gz, file2.gz]`
- Variable expansion: `depends_on: ['!varstr "{lang}.arpa.gz"']`

This ensures the filter step waits for the train_ngram step to complete before starting.

### Running as a SLURM Job

For very long workflows, you can submit the submit/status commands themselves as SLURM jobs:

```bash
# submit_wrapper.sh
#!/bin/bash
#SBATCH --job-name=opusfilter
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

source ~/.bashrc
conda activate opusfilter
opusfilter-slurm-submit config.yaml --workdir /scratch/work
opusfilter-slurm-status /scratch/work --watch
```

Submit with: `sbatch submit_wrapper.sh`

### Resource Usage Collection
Track actual resource usage:

```bash
# After jobs complete
for jobid in $(squeue -u $USER -h | awk '/JobId=/ {print $2}'); do
    seff $jobid
done
```

This enables better resource estimation for future runs.
