# Struktur Data dan Folder

## Folder Utama

| Folder | Deskripsi |
|--------|-----------|
| `Data/` | Menyimpan dataset mentah dan hasil preprocessing |
| `Prepocessing/` | Modul preprocessing Python (dictionary, stopwords) |
| `Output/` | Menyimpan hasil visualisasi (WordCloud, distribusi) |
| `docs/` | Dokumen tesis (proposal, ringkasan analisis) |

## File Utama

| File | Deskripsi |
|------|-----------|
| `01_preprocessing.ipynb` | EDA, preprocessing data, dan split data |
| `02_model_lstm.ipynb` | Model LSTM dengan grid search dan penanganan imbalance |
| `03_model_bilstm.ipynb` | Model BiLSTM (empiris, simulasi, class weight) |
| `04_model_indobertweet_lora.ipynb` | Model IndoBERTweet-LoRA (empiris, simulasi, class weight) |
| `05_analisis_topik.ipynb` | Analisis topik dengan BERTopic, TF-IDF, dan LDA |
| `requirements.txt` | Daftar dependensi Python |

## File Data

| File | Deskripsi |
|------|-----------|
| `Data/data_banjir.csv` | Dataset mentah tweet banjir Sumatera |
| `Data/data_preprocessed_with_emoticon.csv` | Data setelah preprocessing |
| `Data/split_data.pkl` | Data yang sudah di-split train/validation/test |

## File Preprocessing

| File | Deskripsi |
|------|-----------|
| `Prepocessing/emoji_dict.py` | Dictionary konversi emoticon ke teks |
| `Prepocessing/normalisasi_dict.py` | Dictionary normalisasi slang bahasa Indonesia |
| `Prepocessing/stopwords_lstm_processing.py` | Stopwords khusus untuk LSTM (keep_words) |
| `Prepocessing/stopwords_wordcloud.py` | Stopwords khusus untuk WordCloud |
