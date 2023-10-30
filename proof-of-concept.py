import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import datasets
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator

"""
This proof of concept is meant as an example on how sentiment analysis
can be used to analyze users movie reviews to know if they have liked it 
or disliked it. 
"""

# Preprocessing and tokenization
TEXT = Field(tokenize="spacy", lower=True)
LABEL = LabelField(dtype=torch.float)

# Load "Netflix" reviews dataset
# Reviews are stored in a CSV file: id, review, sentiment
train_data, test_data = TabularDataset.splits(
    path="data/",
    train="train.csv",
    test="test.csv",
    format="csv",
    fields=[("id", None), ("text", TEXT), ("label", LABEL)],
    skip_header=True,
)

# Build vocabulary
TEXT.build_vocab(
    train_data, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_
)
LABEL.build_vocab(train_data)

# Create DataLoader
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), batch_size=BATCH_SIZE, device=device
)

# Instantiate the model
VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, _) = self.rnn(embedded)
        return self.fc(hidden[-1])


# Instantiate the model, define loss function and optimizer
model = SentimentModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, PAD_IDX)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())


def train_model(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


N_EPOCHS = 5
for epoch in range(N_EPOCHS):
    train_loss = train_model(model, train_iterator, optimizer, criterion)
    print(f"Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}")

# Testing the model


def predict_sentiment(model, text):
    model.eval()
    with torch.no_grad():
        text = (
            torch.tensor([TEXT.vocab.stoi[word] for word in text])
            .unsqueeze(1)
            .to(device)
        )
        prediction = torch.sigmoid(model(text))
        return prediction.item()


# Test a review
review = "This movie is a waste of time and money."
sentiment = predict_sentiment(model, review)
print(f"Review Sentiment: {sentiment}")
