import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext import datasets
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator

# Preprocessing and tokenization using TorchText
TEXT = Field(tokenize="spacy", lower=True)
LABEL = LabelField(dtype=torch.float)

# Load IMDb dataset
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

# Build vocabulary
TEXT.build_vocab(
    train_data, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_
)
LABEL.build_vocab(train_data)

# Create DataLoader
BATCH_SIZE = 64
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), batch_size=BATCH_SIZE, device=torch.device("cuda")
)

# Define the sentiment analysis model


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


# Instantiate the model
VOCAB_SIZE = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = SentimentModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, PAD_IDX)

# Copy pre-trained embeddings to model
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

# Set the <unk> and <pad> tokens' embeddings to zeros
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# Training the model


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
    print(f"Epoch: {epoch+1:02}")
    print(f"\tTrain Loss: {train_loss:.3f}")

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


# Test a positive review
positive_review = "This movie is fantastic! I loved every moment of it."
positive_sentiment = predict_sentiment(model, positive_review)
print(f"Positive Review Sentiment: {positive_sentiment:.3f}")

# Test a negative review
negative_review = "The movie was terrible. I regretted watching it."
negative_sentiment = predict_sentiment(model, negative_review)
print(f"Negative Review Sentiment: {negative_sentiment:.3f}")
