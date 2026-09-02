# LSTM Context

Architecture:

Input
→ Tokenizer
→ Padding
→ Embedding
→ LSTM
→ Dropout
→ Dense(3)
→ Softmax

Fixed:

* Keras Tokenizer
* OOV token
* post padding
* trainable embedding
* CrossEntropy
* Seed control

Hyperparameter search occurs before balancing experiments.
