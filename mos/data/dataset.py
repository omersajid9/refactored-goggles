import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split


def get_text_dataloader(text_tokenizer, batch_size=64):
    dataset = TextDataset(text_tokenizer)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=dataset.collate_fn)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                collate_fn=dataset.collate_fn)
    return train_dataloader, val_dataloader


class TextDataset(Dataset):
    def __init__(self, tokenizer, seq_len=10):
        super().__init__()

        self.seq_len = seq_len + 1
        self.special_tokens = tokenizer.special_tokens
        self.data = tokenizer.encoded_data
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        for line in self.data:
            line = [self.special_tokens['<|im_start|>']] + line.tolist() + [self.special_tokens['<|im_end|>']]
            for i in range(1, len(line) + self.seq_len - 1):
                start = max(0, i - self.seq_len)
                end = min(i, len(line)-1)
                seq = line[start:end]
                target = line[end]
                samples.append((seq, target))
        return samples

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def collate_fn(self, batch):
        x_batch, y_batch = zip(*batch)
        x_padded = pad_sequence(x_batch, batch_first=True, padding_value=self.special_tokens['<|pad|>'], padding_side="left")
        y_tensor = torch.tensor(y_batch, dtype=torch.long)
        attention_mask = (x_padded != self.special_tokens['<|pad|>']).long()
        return x_padded, y_tensor, attention_mask