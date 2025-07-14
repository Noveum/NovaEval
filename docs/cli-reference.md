---
layout: default
title: CLI Reference
nav_order: 5
---

# CLI Reference

This document provides a comprehensive reference for the NovaEval command-line interface (CLI).

## Table of Contents

- [Installation](#installation)
- [Global Options](#global-options)
- [Commands](#commands)
  - [run](#run)
  - [quick](#quick)
  - [list-datasets](#list-datasets)
  - [list-models](#list-models)
  - [list-scorers](#list-scorers)
  - [generate-config](#generate-config)
- [Examples](#examples)
- [Configuration Files](#configuration-files)
- [Troubleshooting](#troubleshooting)

## Installation

The NovaEval CLI is installed automatically when you install the package:

```bash
pip install novaeval
```

After installation, you can use the CLI with:

```bash
novaeval --help
```

## Global Options

These options are available for all commands:

| Option | Description | Default |
|--------|-------------|---------|
| `--version` | Show version information | - |
| `--log-level` | Set logging level | `INFO` |
| `--log-file` | Specify log file path | - |
| `--help` | Show help message | - |

### Logging Levels

Available logging levels (in order of verbosity):
- `DEBUG`: Detailed debugging information
- `INFO`: General information messages
- `WARNING`: Warning messages only
- `ERROR`: Error messages only
- `CRITICAL`: Critical error messages only

### Example with Global Options

```bash
# Run with debug logging
novaeval --log-level DEBUG run config.yaml

# Save logs to file
novaeval --log-file evaluation.log run config.yaml

# Show version
novaeval --version
```

## Commands

### run

Run an evaluation from a configuration file.

**Usage:**
```bash
novaeval run [OPTIONS] CONFIG_FILE
```

**Arguments:**
- `CONFIG_FILE`: Path to YAML or JSON configuration file (required)

**Options:**
| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--output-dir` | `-o` | Output directory for results | Uses config file setting |
| `--dry-run` | - | Validate configuration without running | `False` |

**Examples:**

```bash
# Basic evaluation
novaeval run config.yaml

# Override output directory
novaeval run -o ./my-results config.yaml

# Validate configuration only
novaeval run --dry-run config.yaml

# With custom logging
novaeval --log-level DEBUG run config.yaml
```

**What it does:**
1. Loads and validates the configuration file
2. Creates an evaluator instance from the configuration
3. Runs the evaluation
4. Saves results to the specified output directory
5. Displays a summary of results

### quick

Quick evaluation with minimal configuration (CLI parameters only).

**Usage:**
```bash
novaeval quick [OPTIONS]
```

**Options:**
| Option | Short | Description | Required | Multiple |
|--------|-------|-------------|----------|----------|
| `--dataset` | `-d` | Dataset name | Yes | No |
| `--model` | `-m` | Model name | Yes | Yes |
| `--scorer` | `-s` | Scorer name | No | Yes |
| `--num-samples` | `-n` | Number of samples | No | No |
| `--output-dir` | `-o` | Output directory | No | No |

**Default Values:**
- `--scorer`: `["accuracy"]`
- `--output-dir`: `"./results"`

**Examples:**

```bash
# Basic quick evaluation
novaeval quick -d mmlu -m gpt-4

# Multiple models and scorers
novaeval quick -d mmlu -m gpt-4 -m claude-3-opus -s accuracy -s f1

# Limited samples
novaeval quick -d mmlu -m gpt-4 -n 100

# Custom output directory
novaeval quick -d mmlu -m gpt-4 -o ./quick-results
```

**Note:** This feature is currently in development and will be available in a future release.

### list-datasets

List all available datasets.

**Usage:**
```bash
novaeval list-datasets
```

**Options:** None

**Example:**
```bash
novaeval list-datasets
```

**Output:**
```
Available Datasets
┌─────────────┬──────────────────────────────────────┐
│ Name        │ Description                          │
├─────────────┼──────────────────────────────────────┤
│ mmlu        │ Massive Multitask Language Understanding │
│ hellaswag   │ Commonsense reasoning                │
│ truthful_qa │ Truthfulness evaluation             │
│ squad       │ Reading comprehension                │
│ glue        │ General Language Understanding       │
└─────────────┴──────────────────────────────────────┘
```

### list-models

List all available model providers.

**Usage:**
```bash
novaeval list-models
```

**Options:** None

**Example:**
```bash
novaeval list-models
```

**Output:**
```
Available Model Providers
┌──────────┬─────────────────────────┬────────────────────────────────┐
│ Provider │ Description             │ Examples                       │
├──────────┼─────────────────────────┼────────────────────────────────┤
│ openai   │ OpenAI GPT models       │ gpt-4, gpt-3.5-turbo          │
│ anthropic│ Anthropic Claude models │ claude-3-opus, claude-3-sonnet │
│ bedrock  │ AWS Bedrock models      │ Various providers              │
│ noveum   │ Noveum AI Gateway       │ Multiple providers             │
└──────────┴─────────────────────────┴────────────────────────────────┘
```

### list-scorers

List all available scorers.

**Usage:**
```bash
novaeval list-scorers
```

**Options:** None

**Example:**
```bash
novaeval list-scorers
```

**Output:**
```
Available Scorers
┌─────────────────────┬────────────────────────────────┐
│ Name                │ Description                    │
├─────────────────────┼────────────────────────────────┤
│ accuracy            │ Classification accuracy        │
│ exact_match         │ Exact string matching          │
│ f1                  │ Token-level F1 score          │
│ semantic_similarity │ Embedding-based similarity    │
│ bert_score          │ BERT-based evaluation         │
│ code_execution      │ Code execution validation     │
│ llm_judge           │ LLM-as-a-judge scoring        │
└─────────────────────┴────────────────────────────────┘
```

### generate-config

Generate a sample configuration file.

**Usage:**
```bash
novaeval generate-config OUTPUT_FILE
```

**Arguments:**
- `OUTPUT_FILE`: Path for the generated configuration file (required)

**Options:** None

**Examples:**

```bash
# Generate YAML configuration
novaeval generate-config sample-config.yaml

# Generate JSON configuration
novaeval generate-config sample-config.json
```

**Generated Configuration:**
```yaml
dataset:
  type: mmlu
  subset: abstract_algebra
  num_samples: 100

models:
  - type: openai
    model_name: gpt-4
    temperature: 0.0

scorers:
  - type: accuracy

output:
  directory: ./results
  formats:
    - json
    - csv
    - html

evaluation:
  max_workers: 4
  batch_size: 1
```

## Examples

### Complete Evaluation Workflow

```bash
# 1. List available datasets
novaeval list-datasets

# 2. List available models
novaeval list-models

# 3. List available scorers
novaeval list-scorers

# 4. Generate a sample configuration
novaeval generate-config my-config.yaml

# 5. Edit the configuration file as needed
# (edit my-config.yaml)

# 6. Validate the configuration
novaeval run --dry-run my-config.yaml

# 7. Run the evaluation
novaeval run my-config.yaml

# 8. Run with custom output directory
novaeval run -o ./evaluation-results my-config.yaml
```

### Development and Testing

```bash
# Debug mode with detailed logging
novaeval --log-level DEBUG run config.yaml

# Save logs for later analysis
novaeval --log-file debug.log --log-level DEBUG run config.yaml

# Quick validation without running
novaeval run --dry-run config.yaml
```

## Configuration Files

The CLI supports both YAML and JSON configuration files. The format is automatically detected from the file extension.

### YAML Example

```yaml
dataset:
  type: mmlu
  subset: abstract_algebra
  num_samples: 100

models:
  - type: openai
    model_name: gpt-4
    temperature: 0.0
  - type: anthropic
    model_name: claude-3-opus
    temperature: 0.0

scorers:
  - type: accuracy
  - type: f1

output:
  directory: ./results
  formats:
    - json
    - csv
    - html

evaluation:
  max_workers: 4
  batch_size: 1
```

### JSON Example

```json
{
  "dataset": {
    "type": "mmlu",
    "subset": "abstract_algebra",
    "num_samples": 100
  },
  "models": [
    {
      "type": "openai",
      "model_name": "gpt-4",
      "temperature": 0.0
    }
  ],
  "scorers": [
    {
      "type": "accuracy"
    }
  ],
  "output": {
    "directory": "./results",
    "formats": ["json", "csv", "html"]
  },
  "evaluation": {
    "max_workers": 4,
    "batch_size": 1
  }
}
```

## Troubleshooting

### Common Issues

**1. Configuration File Not Found**
```bash
Error: Invalid value for 'CONFIG_FILE': File 'config.yaml' does not exist.
```
**Solution:** Ensure the configuration file path is correct and the file exists.

**2. Invalid Configuration**
```bash
Error: Invalid configuration: ...
```
**Solution:** Use `--dry-run` to validate your configuration before running.

**3. Missing API Keys**
```bash
Error: OpenAI API key not found
```
**Solution:** Set the required environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
```

**4. Permission Errors**
```bash
Error: Permission denied when creating output directory
```
**Solution:** Check directory permissions or use a different output directory.

### Getting Help

- Use `--help` with any command for detailed help
- Check the log files for detailed error information
- Use `--log-level DEBUG` for verbose output
- Visit the [GitHub Issues](https://github.com/Noveum/NovaEval/issues) page

### Environment Variables

Common environment variables used by NovaEval:

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | For OpenAI models |
| `ANTHROPIC_API_KEY` | Anthropic API key | For Anthropic models |
| `AWS_ACCESS_KEY_ID` | AWS access key | For Bedrock models |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | For Bedrock models |
| `NOVEUM_API_KEY` | Noveum API key | For Noveum models |

### Performance Tips

1. **Use appropriate batch sizes** in your configuration
2. **Set max_workers** based on your system capabilities
3. **Use --dry-run** to validate configurations quickly
4. **Monitor API usage** when using paid model providers
5. **Use smaller sample sizes** for initial testing

---

For more information, see the [User Guide](user-guide.md) and [API Reference](api-reference.md).
