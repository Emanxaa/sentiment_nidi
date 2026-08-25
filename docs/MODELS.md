# Model dan Algoritma

## Ringkasan Model

Repositori ini menggunakan tiga model untuk klasifikasi sentimen:

| Model | Notebook | Framework | Teknik Penanganan Imbalance |
|-------|----------|-----------|------------------------------|
| LSTM | `02_model_lstm.ipynb` | TensorFlow/Keras | Grid Search + Class Weight + Oversampling + Undersampling + SMOTE |
| BiLSTM | `03_model_bilstm.ipynb` | TensorFlow/Keras | Data Simulasi (1:1:1, 6:3:1, 8:1:1) + Class Weight |
| IndoBERTweet-LoRA | `04_model_indobertweet_lora.ipynb` | PyTorch + Transformers + PEFT | Data Simulasi (1:1:1, 6:3:1, 8:1:1) + Class Weight |

## 1. Model LSTM (Long Short-Term Memory)

### Arsitektur
- **Embedding Layer**: `input_dim=10000`, `output_dim=128`, `input_length=50`
- **LSTM Layer**: 32 atau 64 units (dari grid search)
- **Dropout**: 0.2 atau 0.3
- **Dense Layer**: 64 units (ReLU)
- **Output Layer**: 3 units (Softmax) untuk 3 kelas sentimen

### Hyperparameter Tuning
- Grid search dengan 24 kombinasi hyperparameter
- Batch size: 16, 32
- Learning rate: 5e-5, 1e-4, 2e-4
- Dropout: 0.2, 0.3
- Units: 32, 64

### Penanganan Imbalance
1. **Baseline**: Data asli tanpa penanganan
2. **Class Weight**: Menghitung bobot kelas menggunakan `compute_class_weight`
3. **Random Oversampling**: `RandomOverSampler`
4. **Random Undersampling**: `RandomUnderSampler`
5. **SMOTE**: Synthetic Minority Over-sampling Technique

### Tokenizer
- Keras Tokenizer dengan `num_words=10000`
- OOV token: `<OOV>`
- Padding: `post`, `maxlen=50`

## 2. Model BiLSTM (Bidirectional LSTM)

### Arsitektur
- **Embedding Layer**: `input_dim=10000`, `output_dim=128`, `input_length=50`
- **Bidirectional LSTM**: 64 units
- **Dropout**: 0.3
- **Dense Layer**: 64 units (ReLU)
- **Output Layer**: 3 units (Softmax)

### Optimizer
- Adam dengan `learning_rate=0.001`
- Loss: `sparse_categorical_crossentropy`

### Data Simulasi
Membuat skenario distribusi kelas yang berbeda dari data train:
1. **Skenario 1:1:1**: Seimbang sempurna
2. **Skenario 6:3:1**: Mayoritas negatif, sedang positif, minoritas netral
3. **Skenario 8:1:1**: Mayoritas dominan negatif

### Class Weight
Menghitung bobot kelas dari data train final menggunakan `compute_class_weight` dengan parameter `balanced`.

## 3. Model IndoBERTweet-LoRA

### Base Model
- `indolem/indobertweet-base-uncased`
- Fine-tuned dengan LoRA (Low-Rank Adaptation)

### LoRA Configuration
- **r (rank)**: 8 atau 16
- **lora_alpha**: 16 atau 32
- **target_modules**: `query`, `value`
- **lora_dropout**: 0.2 atau 0.3
- **task_type**: `SEQ_CLS` (Sequence Classification)
- **modules_to_save**: `classifier`

### Arsitektur
- Model dasar: IndoBERTweet-base (12 layers, 768 hidden dim, 12 heads)
- Classifier head: Linear layer untuk 3 kelas
- Hanya LoRA adapter dan classifier head yang di-train

### Hyperparameter Tuning
- Grid search dengan 6 kombinasi
- Batch size: 16, 32
- Learning rate: 5e-5, 1e-4, 2e-4
- Dropout: 0.2, 0.3
- r: 8, 16
- alpha: 16, 32

### Tokenizer
- AutoTokenizer dari `indolem/indobertweet-base-uncased`
- Max length: 128

### Dataset
- Custom `torch.utils.data.Dataset`
- Padding dan truncation otomatis

## 4. Analisis Topik

### BERTopic
- **Embedding Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **UMAP**: `n_neighbors=15`, `n_components=5`, `metric='cosine'`
- **HDBSCAN**: `min_cluster_size=15`, `metric='euclidean'`
- **Vectorizer**: `CountVectorizer` dengan `ngram_range=(1,2)`
- Dilakukan per kelas sentimen (negatif, netral, positif)

### TF-IDF
- `TfidfVectorizer` dengan `max_features=1000`, `ngram_range=(1,2)`
- Dilakukan per kelas sentimen
- Ambil top 20 kata kunci per sentimen

### LDA (Latent Dirichlet Allocation)
- `LatentDirichletAllocation` dengan `n_components=3`
- `CountVectorizer` dengan `max_features=1000`, `ngram_range=(1,2)`
- Dilakukan per kelas sentimen
- Coherence score dihitung menggunakan `gensim` CoherenceModel
