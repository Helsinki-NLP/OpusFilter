# OpusFilter SLURM Integration

OpusFilter now supports running workflows on SLURM clusters with resource-optimized job scheduling.

## Overview

The `opusfilter-slurm` command converts OpusFilter workflows into SLURM jobs, handling:
- Automatic dependency management
- Per-step resource allocation
- Concurrent execution of independent steps
- Job monitoring and status tracking
- Resume capability

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

2. Run the workflow:

```bash
# Basic execution
opusfilter-slurm config.yaml

# With options
opusfilter-slurm config.yaml \
    --resume \
    --max-concurrent 10 \
    --workdir /scratch/myproject-work
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

- Independent steps run simultaneously (up to `--max-concurrent`)
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

1. **Estimate Resources**
   - Start with conservative time/memory limits
   - Check actual usage with `seff` after completion
   - Adjust based on historical data

2. **Organize Workflows**
   - Place I/O-intensive steps early (opus_read, concatenate)
   - Group similar resource requirements
   - Avoid unnecessary dependencies

3. **Use Arrays**
   - Enable array_size for filter/score steps
   - Parallelizes within step, not just between steps

4. **Monitor Progress**
   - Check logs in `${workdir}/logs/`
   - Set up email notifications

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

## Integration with Other Tools

The SLURM integration outputs standard OpusFilter files that can be used with:
- `opusfilter-diagram`: Visualize workflow
- `opusfilter-scores`: Analyze job scores
- Custom monitoring scripts

## Advanced Usage

### Custom Job Dependencies
For complex workflows, explicitly specify dependencies:

```yaml
steps:
  - type: step1
    parameters:
      outputs: [file1]
      depends_on: [setup]  # Explicit dependency
      
  - type: step2
    parameters:
      inputs: [file1]
```

### Resource Usage Collection
Track actual resource usage:

```bash
# After jobs complete
for jobid in $(squeue -u $USER -h | awk '/JobId=/ {print $2}'); do
    seff $jobid
done
```

This enables better resource estimation for future runs.
