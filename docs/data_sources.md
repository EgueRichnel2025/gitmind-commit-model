# Training Data Sources

## Objective

Build a dataset for supervised Git commit message generation.

Expected format:
- Input: Git diff.
- Target: Commit message.

## Primary dataset: CommitChronicle

Repository:
JetBrains-Research/commit-chronicle

Characteristics:
- Approximately 10.9 million commits.
- Approximately 11,900 repositories.
- 20 programming languages.
- Parquet format.
- Train, validation and test splits.

Important fields:
- repo
- hash
- message
- original_message
- language
- license
- mods

The dataset was collected from repositories using permissive
licenses, including MIT, Apache 2.0 and BSD 3-Clause.

The original license information will be retained.
An automated filter will select eligible examples.

## Excluded dataset: CommitBench

Repository:
Maxscha/commitbench

CommitBench is excluded from the current experiment
to simplify dataset management.

## Data preparation

The pipeline will:
1. Load the dataset in streaming mode.
2. Inspect its structure.
3. Filter examples using repository license metadata.
4. Remove incomplete and duplicate examples.
5. Reconstruct Git diffs.
6. Prepare training, validation and test datasets.

## Current decision

CommitChronicle is the only external dataset selected
for the initial experiment.
