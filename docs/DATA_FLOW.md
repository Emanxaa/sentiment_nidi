# Alur Data dan Pipeline

## Alur Data Keseluruhan

```
                        +-------------------+
                        |   data_banjir.csv |
                        |  (Dataset Mentah) |
                        +--------+----------+
                                 |
                                 v
                        +-------------------+
                        |   01_preprocessing |
                        |   (EDA & Split)    |
                        +--------+----------+
                                 |
               +-----------------+-----------------+
               |                 |                 |
               v                 v                 v
        +-------------+   +--------------+   +--------------+
        | WordCloud    |   |  Data Split  |   |  EDA Stats   |
        | Output       |   |  (pickle)    |   |              |
        +-------------+   +------+-------+   +--------------+
                                 |
                                 v
                        +-------------------+
                        | data_preprocessed |
                        |  .with_emoticon   |
                        |      .csv         |
                        +--------+----------+
                                 |
                +----------------+----------------+
                |                |                |
                v                v                v
        +--------------+ +--------------+ +--------------+
        | 02_model_lstm | |03_model_bilstm| |04_model_bert |
        |               | |               | |              |
        +-------+------+ +-------+------+ +------+-------+
                |                |                |
                v                v                v
        +--------------+ +--------------+ +--------------+
        | Hasil LSTM   | | Hasil BiLSTM | | Hasil Indo-  |
        | (CSV)        | | (CSV)        | | BERTweet-LoRA|
        +--------------+ +--------------+ +--------------+
                                 |
                                 v
                        +-------------------+
                        |  05_analisis_topik |
                        | (BERTopic, TF-IDF, |
                        |     LDA)           |
                        +-------------------+
```

## Detail Pipeline per Model

### 1. Pipeline LSTM

```
split_data.pkl
       |
       v
+-------------------+
| Tokenizer Keras   |  ----> Vocabulary size: 10,000, OOV: <OOV>
+-------------------+
       |
       v
+-------------------+
| Padding Sequences |  ----> Max length: 50, padding: post
+-------------------+
       |
       v
+-------------------+
| Imbalance Handling|  ----> Baseline / Class Weight / Oversampling / Undersampling / SMOTE
+-------------------+
       |
       v
+-------------------+
| Grid Search       |  ----> 5 strategis x 24 hyperparameter = 120 eksperimen
+-------------------+
       |
       v
+-------------------+
| Model LSTM        |  ----> Embedding -> LSTM -> Dropout -> Dense -> Output
+-------------------+
       |
       v
+-------------------+
| Training          |  ----> EarlyStopping (patience=3), restore_best_weights
+-------------------+
       |
       v
+-------------------+
| Evaluasi          |  ----> Classification Report, Confusion Matrix, Macro F1
+-------------------+
```

### 2. Pipeline BiLSTM

```
split_data.pkl
       |
       v
+-------------------+
| Data Simulasi     |  ----> Skenario 1:1:1 / 6:3:1 / 8:1:1
+-------------------+
       |
       v
+-------------------+
| Train-Val Split   |  ----> Stratified split (90:10)
+-------------------+
       |
       v
+-------------------+
| Tokenizer Keras   |  ----> Vocabulary size: 10,000, OOV: <OOV>
+-------------------+
       |
       v
+-------------------+
| Padding Sequences |  ----> Max length: 50
+-------------------+
       |
       v
+-------------------+
| Model BiLSTM      |  ----> Embedding -> Bidirectional(LSTM) -> Dropout -> Dense -> Output
+-------------------+
       |
       v
+-------------------+
| Training          |  ----> EarlyStopping (patience=3)
+-------------------+
       |
       v
+-------------------+
| Evaluasi Test     |  ----> Classification Report, Confusion Matrix
+-------------------+
```

### 3. Pipeline IndoBERTweet-LoRA

```
split_data.pkl
       |
       v
+-------------------+
| Data Simulasi     |  ----> Skenario 1:1:1 / 6:3:1 / 8:1:1
+-------------------+
       |
       v
+-------------------+
| Tokenizer BERT    |  ----> IndoBERTweet tokenizer, max_length=128
+-------------------+
       |
       v
+-------------------+
| PyTorch Dataset   |  ----> SentimenDataset class
+-------------------+
       |
       v
+-------------------+
| Model LoRA        |  ----> IndoBERTweet + LoRA Adapter + Classifier
+-------------------+
       |
       v
+-------------------+
| Hyperparameter    |  ----> Grid Search (6 kombinasi)
| Tuning            |
+-------------------+
       |
       v
+-------------------+
| Trainer HF        |  ----> TrainingArguments, WeightedTrainer (untuk class weight)
+-------------------+
       |
       v
+-------------------+
| Evaluasi Test     |  ----> Classification Report, Confusion Matrix
+-------------------+
```

## Algoritma yang Digunakan

### Preprocessing
1. **Regex Cleaning**: Pattern matching untuk URL, mention, hashtag, noise
2. **Tokenization**: Word-based splitting (LSTM), BPE/WordPiece (BERT)
3. **Stemming**: Sastrawi stemmer untuk bahasa Indonesia
4. **Normalisasi**: Dictionary-based slang normalization

### Modeling
1. **LSTM**: Recurrent neural network untuk sequence data
2. **BiLSTM**: Bidirectional LSTM untuk konteks dua arah
3. **LoRA (Low-Rank Adaptation)**: Fine-tuning efisien untuk model besar
4. **Attention Mechanism**: Built-in dalam BERT (self-attention)

### Evaluation
1. **Classification Report**: Precision, Recall, F1-score per kelas
2. **Confusion Matrix**: Matriks prediksi vs aktual
3. **Macro F1**: Rata-rata F1-score (tidak dipengaruhi oleh imbalance)
4. **Accuracy**: Persentase prediksi benar

### Topic Modeling
1. **BERTopic**: Topic modeling berbasis embedding dan clustering
2. **TF-IDF**: Term Frequency-Inverse Document Frequency
3. **LDA**: Latent Dirichlet Allocation
4. **UMAP**: Dimensionality reduction untuk visualisasi
5. **HDBSCAN**: Density-based clustering

## Label Encoding

| Sentimen | Label Integer |
|----------|---------------|
| Negatif  | 0             |
| Netral   | 1             |
| Positif  | 2             |

## Split Data

- **Train**: 80% dari total data
- **Validation**: 10% dari train data (stratified)
- **Test**: 20% dari total data (dipertahankan untuk evaluasi akhir)
- **Stratifikasi**: Berdasarkan label sentimen untuk menjaga distribusi
