from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

class TextDataset(Dataset):
    def __init__(self, encoder, seq_len=10, train=True, train_split=0.8):


class TextDataset(Dataset):
    def __init__(self, encoded_lines, pad_idx, eos_idx, start_idx, seq_len=10, train=True, train_split=0.8):
        super().__init__()
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.seq_len = seq_len
        self.start_idx = start_idx

        split_idx = int(len(encoded_lines) * train_split)
        self.data = encoded_lines[:split_idx] if train else encoded_lines[split_idx:]
        self.pairs = self._get_sequential_pairs()

    def _get_sequential_pairs(self):
        pairs = []
        for line in self.data:
            line_with_eos = [self.start_idx] + line + [self.eos_idx]
            for i in range(1, len(line_with_eos)):
                start_idx = max(0, i - self.seq_len + 1)
                X = line_with_eos[start_idx:i]
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
        Xs_padded = pad_sequence(Xs, batch_first=True, padding_value=self.pad_idx)
        ys_padded = pad_sequence(ys, batch_first=True, padding_value=self.pad_idx)
        lengths = torch.tensor([len(x) for x in Xs])
        return Xs_padded, ys_padded, lengths

