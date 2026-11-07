# Training Data Sources

## Objective

Identify and evaluate datasets for supervised Git commit message
generation.

The expected training format is:

- Input: Git diff.
- Target: commit message.

## Primary candidate: CommitChronicle

Repository:
JetBrains-Research/commit-chronicle

Dataset characteristics:

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

The mods field contains file modifications.
Each modification can contain its own Git diff.

Preprocessing will reconstruct a combined diff for each commit.

### License policy

The dataset contains source repositories under different licenses.

The license field must be retained for every example.
No example will enter the training dataset before license review.

## Secondary candidate: CommitBench

Repository:
Maxscha/commitbench

Dataset characteristics:

- 1,664,590 examples according to its dataset card.
- Six programming languages.
- CSV format.
- Direct diff/message pairs.

Dataset license: CC BY-NC 4.0.

### License policy

CommitBench will not be included in the default training dataset.

Its use requires compliance with the non-commercial restriction
and appropriate attribution.

## Data quality requirements

The exploration pipeline must examine:

- Missing values.
- Duplicate commits.
- Empty diffs.
- Empty messages.
- Diff length distribution.
- Message length distribution.
- Programming language distribution.
- Repository distribution.
- License distribution.
- Sensitive information.

## Data splitting

Repository-level separation must be preserved.

The same repository must not leak between training,
validation and test datasets.

Existing official splits will be inspected before use.

## Current decision

CommitChronicle is the primary candidate.
CommitBench is reserved for restricted experimentation.

No large-scale download or model training is authorized
by this preliminary inventory.

## CommitBench: non-commercial usage

GitMind is currently developed for personal, educational and
non-commercial purposes.

CommitBench may therefore be used for non-commercial experiments
under CC BY-NC 4.0.

Attribution and license notices must be preserved.
Any modifications must be documented.

Commercial reuse requires a separate license review.
Public distribution of trained model weights also requires
a review of the applicable rights and conditions.
