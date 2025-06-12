import torch.nn as nn

class Model(nn.Module):
    def __init__(self, vocab_size, embed_size = 126, hidden_size = 256, num_layers = 1, dropout = 0.5):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.rnn_dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedd = self.embedding_layer(x)
        output, hidden = self.rnn(embedd, hidden)
        output = self.rnn_dropout(output)
        y_hat = self.linear(output)
        return y_hat, hidden
    
    def forward(self, x, hidden=None):
        # print(f"X INPUT SHAPE: {x.shape}")
        embedded = self.embedding_layer(x)
        # print(f"EMBED OUTPUT SHAPE: {embedded.shape}")
        # output, hidden = self.rnn(embedd, hidden)
        rnn_out, hidden = self.rnn(embedded)
        # print(f"MODEL OUTPUT SHAPE: {rnn_out.shape}")
        # print(f"MODEL HIDDEN SHAPE: {hidden.shape}")
        rnn_out = hidden[:, -1, :]
        output = self.rnn_dropout(rnn_out)
        y_hat = self.linear(output)
        return y_hat, hidden