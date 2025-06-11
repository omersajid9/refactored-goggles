import torch.nn as nn

class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, hidden_size = 256, num_layers = 3, dropout = 0.5):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.lstm_dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedd = self.embedding_layer(x)
        output, hidden = self.lstm(embedd, hidden)
        output = self.lstm_dropout(output)
        y_hat = self.linear(output)
        # y_hat = self.softmax(y_hat)
        return y_hat, hidden
