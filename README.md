# GitMind Commit Model

A machine learning project dedicated to generating meaningful Git commit
messages from code diffs.

## Objective

Develop and fine-tune a specialized AI model capable of understanding
code modifications and generating relevant commit messages.

## Technical approach

- Task: Conditional text generation.
- Learning paradigm: Supervised learning.
- Architecture: Transformer Encoder-Decoder.
- Initial model candidate: Salesforce CodeT5-small.
- Framework: PyTorch and Hugging Face Transformers.
- Training environment: Google Colab.
- Inference: Local execution through GitMind CLI.

## Machine Learning Pipeline

1. Collect commit datasets.
2. Clean and analyze the data.
3. Prepare training, validation and test datasets.
4. Load a pretrained model.
5. Fine-tune the model.
6. Evaluate its performance.
7. Export the trained model.
8. Integrate it into GitMind.

## Repository Structure

- configs/: model and training configurations.
- data/: datasets.
- notebooks/: experiments and Colab notebooks.
- scripts/: data preparation scripts.
- src/: Python source code.
- tests/: automated tests.
- docs/: technical documentation.
- artifacts/: generated model artifacts.

## Data Governance

External datasets must be checked for their licenses and usage conditions.
Private source code, credentials and sensitive information must never be
included without authorization.

## Project Status

Initial development: repository structure and development tooling.
