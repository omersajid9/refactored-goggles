import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class TextDataset(Dataset):
    vocab_size: int
    def __init__(self, file_path, seq_len=100, train=True, train_split=0.8):
        super().__init__()

        self.seq_len = seq_len

        with open(file_path, "r", encoding="UTF-8") as file:
            self.content = file.read()

        self.chars = list(set(self.content))
        self.vocab_size = len(self.chars)

        self.char_to_indx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

        self.data = [self.char_to_indx[char] for char in self.content]

        split_idx = int(train_split*len(self.data))

        if train:
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]

        print(f"Dataset initialized:")
        print(f"- Vocabulary size: {self.vocab_size}")
        print(f"- Data length: {len(self.data)}")
        print(f"- Mode: {'Train' if train else 'Validation'}")

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx + self.seq_len]
        target_seq = self.data[idx + 1:idx + self.seq_len + 1]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


class TextPredictor(nn.Module):
    
    def __init__(self, vocab_size, hidden_size=512, num_layers=2, dropout=0.5):
        super().__init__()

        # (vocab_size, hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            # batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size, vocab_size)

    """
        Inputs:
            - x: (batch_size, seq_len)
        Outputs:
            - y: (batch_size, seq_len, vocab_size)
    """
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output, hidden


batch_size, seq_len = 32, 100

if __name__ == "__main__":

    train_dataset = TextDataset("./paul_graham_essays.txt", seq_len=seq_len, train=True)
    val_dataset = TextDataset("./paul_graham_essays.txt", seq_len=seq_len, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    vocab_size = train_dataset.vocab_size

    model = TextPredictor(vocab_size=vocab_size, hidden_size=512, num_layers=2, dropout=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created:")
    print(f"- Vocabulary size: {vocab_size}")
    print(f"- Hidden size: 512")
    print(f"- Number of layers: 2")
    print(f"- Dropout: 0.5")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,}")

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        output, hidden = model(dummy_input)
        print(f"\nForward pass test:")
        print(f"- Input shape: {dummy_input.shape}")
        print(f"- Output shape: {output.shape}")
        print(f"- Hidden state shapes: {hidden[0].shape}, {hidden[1].shape}")
    
    # Parameter breakdown
    print(f"\nParameter breakdown:")
    for name, param in model.named_parameters():
        print(f"- {name}: {param.shape} ({param.numel():,} params)")



    # for batch_idx, (inputs, targets) in enumerate(train_loader):
    #     print(f"Batch {batch_idx}:")
    #     print(f"Input shape: {inputs.shape}")   # [batch_size, seq_len]
    #     print(f"Target shape: {targets.shape}") # [batch_size, seq_len]
        
    #     # Show first sequence in batch as characters
    #     first_input = inputs[0]
    #     first_target = targets[0]
        
    #     input_text = ''.join([train_dataset.idx_to_char[idx.item()] for idx in first_input])
    #     target_text = ''.join([train_dataset.idx_to_char[idx.item()] for idx in first_target])
        
    #     print(f"Input text (first 50 chars): {input_text[:50]}...")
    #     print(f"Target text (first 50 chars): {target_text[:50]}...")
    #     break

