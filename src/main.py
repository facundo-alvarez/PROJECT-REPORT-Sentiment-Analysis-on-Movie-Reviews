import torch
import csv
import spacy
import re
import torch.nn.functional as F
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
                self.labels.append(1 if row[1] == 'positive' else 0)

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        clean_text = re.sub(r"<.*?>", "", review)
        review_doc = self.nlp(clean_text)
        review_doc = [token.lemma_.lower() for token in review_doc if token.is_alpha and not token.is_stop]

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

def collate_fn(batch, max_len=100):
    reviews, labels = zip(*batch)
    reviews_idx = []
    for r in reviews:
        idxs = [vec.stoi.get(tok, 0) for tok in r]
        if len(idxs) < max_len:
            idxs += [0] * (max_len - len(idxs))  # pad
        else:
            idxs = idxs[:max_len]
        reviews_idx.append(idxs)
    reviews_tensor = torch.tensor(reviews_idx, dtype=torch.long)
    labels_tensor = torch.tensor([int(l) for l in labels], dtype=torch.long)
    return reviews_tensor, labels_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mrd = MovieReviewsDataset(r".\data\IMDB Dataset.csv")

g = torch.manual_seed(49)
train_set, validation_set, test_set = random_split(mrd, [0.7, 0.15, 0.15], generator=g)

train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True, collate_fn=collate_fn)
validation_dataloader = DataLoader(validation_set, batch_size=128, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_set, batch_size=128, shuffle=False, collate_fn=collate_fn)

embedding_dim = 300
vec = GloVe(name="6B", dim=embedding_dim)
model = SentimentCNN(embedding_dim, num_classes=2).to(device)
model.embedding = nn.Embedding.from_pretrained(vec.vectors, freeze=False).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 100
best_val_loss = float("inf")
save_path = "sentiment_cnn.pt"

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_dataloader, 0):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (i+1):.3f}')
        running_loss = 0.0

    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in validation_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(validation_dataloader)
    val_acc = correct / total

    print(f"Epoch {epoch+1}/{epochs} "
          f"Train Loss: {running_loss/len(train_dataloader):.4f} "
          f"Val Loss: {avg_val_loss:.4f} "
          f"Val Acc: {val_acc:.4f}")

    # ---- Save best model ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print(f"Model improved. Saved to {save_path}")