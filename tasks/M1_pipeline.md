# Milestone M1 — Build Reusable LSTM Pipeline

Read:

* AGENT.md
* CONTEXT/PROJECT.md
* CONTEXT/LSTM.md
* CONTEXT/KAGGLE.md

Implement only this milestone.

## Goal

Build the reusable LSTM training pipeline.

No balancing experiments yet.

## Input

Data/processed/banjir_processed_v2.csv

## Deliverables

Create:

* utils/data_loader.py
* utils/tokenizer.py
* utils/model_lstm.py
* utils/trainer.py
* utils/evaluator.py
* configs/lstm_config.yaml
* experiments/run_lstm.py

## Required behavior

Implement:

* 80/20 stratified split
* Validation from Train
* Tokenizer
* Padding
* Trainable embedding
* LSTM model
* Early stopping
* Best checkpoint loading
* GPU training on Tesla T4
* Mixed precision

## Output

Running

python experiments/run_lstm.py

must:

* split data
* train baseline LSTM
* save best checkpoint
* generate history.csv
* generate metrics.json

Do not implement balancing.

Stop after M1.
