import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split


def get_text_dataloader(text_tokenizer, batch_size=16):
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
    def __init__(self, tokenizer, seq_len=4):
        super().__init__()

        self.seq_len = seq_len + 1
        self.special_tokens = tokenizer.special_tokens
        self.data = tokenizer.encoded_data
        self.pairs = self._get_sequential_pairs()

    def _get_sequential_pairs(self):
        pairs = []
        for line in self.data:
            line_with_eos = [self.special_tokens['<|im_start|>']] + line.tolist() + [self.special_tokens['<|im_end|>']]
            for i in range(1, len(line_with_eos)):
                start_idx = max(0, i - self.seq_len + 1)
                X = line_with_eos[start_idx:i]
                y = line_with_eos[i]
                y = line_with_eos[start_idx + 1:i + 1]
                pairs.append((X, y))
        return pairs

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        X = torch.tensor(pair[0], dtype=torch.long)
        y = torch.tensor(pair[1], dtype=torch.long)
        return X, y

    def __len__(self):
        return len(self.pairs)

    def collate_fn(self, batch):
        Xs, ys = zip(*batch)
        Xs_padded = pad_sequence(Xs, batch_first=True, padding_value=self.special_tokens['<|pad|>'])
        ys_padded = pad_sequence(ys, batch_first=True, padding_value=self.special_tokens['<|pad|>'])
        lengths = torch.tensor([len(x) for x in Xs])
        return Xs_padded, ys_padded, lengths