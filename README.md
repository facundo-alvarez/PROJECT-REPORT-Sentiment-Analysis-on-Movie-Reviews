# Sentiment Analysis on Movie Reviews

Binary sentiment classification (positive / negative) on the IMDB 50k movie-review
dataset, using a convolutional neural network over pretrained word embeddings.

## Features

- **Full NLP preprocessing pipeline** — HTML tag removal, lowercasing, tokenization,
  stopword removal, **POS-aware lemmatization**, and padding/truncation to a fixed length.
- **Pretrained embeddings** — GloVe 6B (300-dimensional), loaded into a trainable
  embedding layer.
- **TextCNN architecture** — parallel 1D convolutions with kernel sizes 3, 4, and 5
  (n-gram features), global max-pooling, concatenation, dropout, and a fully connected
  classifier.
- **Thorough evaluation** — classification report (precision / recall / F1), confusion
  matrix (raw and normalized), and ROC curve with AUC.

## Tech stack

- Python, PyTorch
- NLTK (tokenization, stopwords, lemmatization, POS tagging)
- torchtext (GloVe embeddings)
- scikit-learn, matplotlib, seaborn (evaluation & visualization)

## Getting started

```bash
pip install -r requirements.txt
```

Download the IMDB dataset of 50k movie reviews and place `IMDB Dataset.csv` under
`data/`. Then open `main.ipynb` and run the cells in order.

> The dataset is available on Kaggle:
> `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews`

## Pipeline overview

```
raw reviews
   │  clean (HTML, lowercase) → tokenize → remove stopwords → lemmatize → pad/truncate
   ▼
GloVe 300d embeddings  →  TextCNN (kernels 3/4/5 + max-pool)  →  positive / negative
```

> Note: A portfolio project. The training cell ships with a low epoch count for quick
> runs — increase epochs before reporting final metrics.
