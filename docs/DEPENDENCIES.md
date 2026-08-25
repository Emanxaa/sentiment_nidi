# Dependensi dan Library

## Library Utama

### Deep Learning & Machine Learning
- **TensorFlow**: Framework untuk model LSTM dan BiLSTM
- **PyTorch**: Framework untuk model IndoBERTweet-LoRA
- **Transformers**: Library Hugging Face untuk model pretrained (BERT)
- **PEFT (Parameter-Efficient Fine-Tuning)**: Library untuk LoRA

### NLP & Text Processing
- **Sastrawi**: Stemmer dan stopword remover bahasa Indonesia
- **Sentence Transformers**: Embedding model untuk BERTopic
- **Gensim**: Library untuk LDA dan coherence score

### Topic Modeling
- **BERTopic**: Topic modeling berbasis embedding
- **UMAP-learn**: Dimensionality reduction
- **HDBSCAN**: Density-based clustering

### Data Processing & Visualization
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Matplotlib**: Plotting dasar
- **Seaborn**: Visualisasi statistik
- **Plotly**: Visualisasi interaktif
- **WordCloud**: Visualisasi frekuensi kata
- **Scikit-learn**: Machine learning utilities (train_test_split, metrics, TF-IDF, LDA)
- **Imbalanced-learn**: Penanganan data tidak seimbang

### Hyperparameter Tuning
- **Optuna**: (Tercantum di requirements, mungkin untuk eksperimen lain)

## Versi atau Catatan
- Menggunakan Python 3.11/3.13 (berdasarkan cache `__pycache__`)
- Mendukung CUDA untuk pelatihan GPU pada model IndoBERTweet-LoRA
