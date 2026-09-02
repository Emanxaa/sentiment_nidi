# IndoBERT-LoRA Context

## Objective

Fine-tune IndoBERTweet using LoRA for Indonesian flood sentiment classification.

The goal is to compare the transformer model fairly against the completed LSTM experiment suite.

## Model

* Base model: `indolem/indobertweet-base-uncased`
* Framework: Hugging Face Transformers
* PEFT LoRA

## Dataset

Use:

`Data/processed/banjir_processed_v2.csv`

The same dataset used in LSTM experiments.

## Split

Reuse the identical protocol.

* Train 80%
* Test 20%
* Stratified
* Validation = 10% of Train

## Evaluation

Primary metric:

* Macro F1

Secondary metrics:

* Accuracy
* Precision
* Recall

Generate for every experiment:

* confusion_train.png
* confusion_val.png
* confusion_test.png
* classification_report.csv
* loss_curve.png
* accuracy_curve.png

## Rules

* Never modify raw data.
* Never rebalance Validation or Test.
* Tokenize only after splitting.
* Restore the best checkpoint before testing.
* Every experiment uses the same evaluation protocol as LSTM.
