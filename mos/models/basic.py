import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, vocab_size, embed_size = 126, hidden_size = 256, num_layers = 3, dropout = 0.5):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.rnn_norm = nn.LayerNorm(hidden_size)
        self.rnn_dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, attention_mask=None, hidden=None):
        embedded = self.embedding_layer(x)
        rnn_out, hidden = self.rnn(embedded, hidden)
        rnn_out = self.rnn_norm(rnn_out)
        rnn_out = self.rnn_dropout(rnn_out)
        rnn_out = rnn_out[:, -1, :]
        logits = self.linear(rnn_out)
        return logits, hidden