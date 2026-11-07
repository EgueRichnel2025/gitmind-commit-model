# GitMind Commit Model - Preprocessing Strategy

## Objective

Prepare CommitChronicle records for supervised commit message generation.

## Source

JetBrains-Research/commit-chronicle.

## Input

A commit may contain multiple modified files.

Each file modification contains:

- old_path
- new_path
- change_type
- diff

The preprocessing pipeline reconstructs a structured representation
for each modified file.

## Target

The processed `message` field is used as the initial training target.

The `original_message` field is retained only for dataset analysis.

## License filtering

The initial experiment accepts the following license labels:

- Apache License 2.0
- MIT License
- BSD 3-Clause New or Revised License

Examples with unknown or unapproved labels are excluded.

## Long diff strategy

CodeT5-small uses an initial source token budget of 512 tokens.

A commit is divided into file-level sections.

Each file section is tokenized and divided into overlapping windows.

The initial overlap is 32 tokens.

Every generated window retains:

- File path
- Change type
- Diff content
- Repository identifier
- Commit hash
- Language
- License
- Global commit message

## Experimental supervision strategy

Each chunk inherits the global commit message of its source commit.

This is a weak-supervision strategy.

It is not assumed to be optimal:
some individual chunks may not independently explain the complete commit message.

Its usefulness must be measured experimentally.

## Future inference strategy

GitMind will preserve the complete Git diff.

The inference pipeline will process all chunks and will later combine
their predictions into a final commit message.

The aggregation strategy is not considered validated until it has
been evaluated on held-out repositories.

## Initial limits

- Source: 512 tokens
- Target: 128 tokens
- Chunk overlap: 32 tokens

These are experimental values, not final hyperparameters.
