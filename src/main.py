import torch
import os
import csv
import spacy
import re
import torch.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchtext.vocab import GloVe

class MovieReviewsDataset(Dataset):
    def __init__(self, dataset_file):
        self.reviews = []
        self.labels = []
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

        with open(dataset_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader, None)
            for row in reader:
                self.reviews.append(row[0])
                self.labels.append(row[1])

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        clean_text = re.sub(r"<.*?>", "", review)
        review_doc = self.nlp(clean_text)
        review_doc = [
            token.text for token in review_doc if 
            not token.is_stop and 
            not token.is_punct and 
            not token.is_space and
            not token.is_digit and
            token.is_alpha and 
            not token.is_bracket and
            not token.is_currency and
            not token.is_quote
        ]
        
        return review_doc, self.labels[idx]
        
class SentimentCNN(nn.Module):
    def __init__(self, embedding_dim, num_classes=2):
        super().__init__()
        
        # 1D CNN layers with different kernel sizes (n-grams)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=100, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=embedding_dim, out_channels=100, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=embedding_dim, out_channels=100, kernel_size=5)
        
        # Fully connected layer
        self.fc = nn.Linear(3 * 100, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Embed words: (batch_size, seq_len, embedding_dim)
        x = self.embedding(x)
        
        # CNN expects (batch_size, embedding_dim, seq_len)
        x = x.permute(0, 2, 1)
        
        # Apply convolution + ReLU + global max pooling
        x1 = F.relu(self.conv1(x)).max(dim=2)[0]
        x2 = F.relu(self.conv2(x)).max(dim=2)[0]
        x3 = F.relu(self.conv3(x)).max(dim=2)[0]
        
        # Concatenate pooled outputs
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.dropout(x)
        
        # Fully connected layer
        logits = self.fc(x)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mrd = MovieReviewsDataset(r".\data\IMDB Dataset.csv")

g = torch.manual_seed(49)
train_set, validation_set, test_set = random_split(mrd, [0.7, 0.15, 0.15], generator=g)

train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_set, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_set, batch_size=64, shuffle=False)

review, label = mrd.__getitem__(1)



embedding_dim = 300
vec = GloVe(name="6B", dim=embedding_dim)
ret = vec.get_vecs_by_tokens(review)
print(ret)