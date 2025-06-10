from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

class TextPipeline:
    @staticmethod
    def generate_dataset(file_path: str, train: bool):
        batch_size = 32

        text_extractor = TextExtractor(file_path)
        lines = text_extractor.get_lines()

        print('Number of lines ', len(lines))

        text_encoder = TextEncoder(lines)
        encoded_lines = text_encoder.encode_lines(lines)

        print('Vocab size ', text_encoder.vocab_size)


        dataset = TextDataset(encoded_lines, text_encoder.pad_idx, text_encoder.eos_idx, train=train)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train,
                                collate_fn=dataset.collate_fn)

        return {
            'dataloader': dataloader,
            'encoder': text_encoder
        }

class TextDataset(Dataset):
    def __init__(self, encoded_lines, pad_idx, eos_idx, seq_len=5, train=True, train_split=0.8):
        super().__init__()
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.seq_len = seq_len

        split_idx = int(len(encoded_lines) * train_split)
        self.data = encoded_lines[:split_idx] if train else encoded_lines[split_idx:]
        self.pairs = self._get_sequential_pairs()

    def _get_sequential_pairs(self):
        pairs = []
        for line in self.data:
            line_with_eos = [self.eos_idx] + line + [self.eos_idx]
            for i in range(len(line_with_eos)):
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
        return len(self.pairs) - self.seq_len

    def collate_fn(self, batch):
        Xs, ys = zip(*batch)
        Xs_padded = pad_sequence(Xs, batch_first=True, padding_value=1)
        ys_padded = pad_sequence(ys, batch_first=True, padding_value=1)
        lengths = torch.tensor([len(x) for x in Xs])
        return Xs_padded, ys_padded, lengths


# Reads a file to extract text
class TextExtractor:
    file_path: str
    content: str
    content_length: int

    def __init__(self, file_path, max_content = -1) -> None:
        self.file_path = file_path
        self.max_content = max_content
        self._load_contents_from_file()

    def _load_contents_from_file(self) -> None:
        with open(self.file_path, "r", encoding="UTF-8") as file:
            content = file.read()
            self.content = content[:self.max_content]

    def get_lines(self) -> list[str]:
        return self.content.strip().split('\n')


# encoding/decoding for text
class TextEncoder:
    def __init__(self, text_content) -> None:
        content = "".join(text_content)
        self.vocab_chars, self.vocab_size = self._get_vocab(content)
        self.char_to_idx, self.idx_to_char = self._get_encoding_and_decoding(self.vocab_chars)
        self.pad_idx = self.char_to_idx.get('<PAD>', len(self.char_to_idx))
        self.eos_idx = self.char_to_idx.get('<EOS>', len(self.char_to_idx))

    def _get_vocab(self, text_content):
        vocab_chars = sorted(list(set(text_content)) + ['<PAD>', '<EOS>'])
        return vocab_chars, len(vocab_chars)
        
    def _get_encoding_and_decoding(self, vocab_chars):
        char_to_idx = {c: i for i, c in enumerate(vocab_chars)}
        idx_to_char = {i: c for i, c in enumerate(vocab_chars)}
        return char_to_idx, idx_to_char
    
    def encode_lines(self, lines):
        return [self.encode(line) for line in lines]
    
    def encode(self, text_content):
        return [self.char_to_idx[char] for char in text_content]
    
    def decode(self, encoded_content):
        return "".join([self.idx_to_char[idx.item()] for idx in encoded_content])
