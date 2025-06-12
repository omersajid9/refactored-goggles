from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import tiktoken

class TextPipeline:
    @staticmethod
    def generate_dataset(file_path: str, train: bool):
        batch_size = 64

        text_extractor = TextExtractor(file_path)
        lines = text_extractor.get_lines()

        print('Number of lines ', len(lines))

        text_encoder = TextEncoder(lines)
        encoded_lines = text_encoder.encode_lines(lines)

        print('Vocab size ', text_encoder.vocab_size)


        dataset = TextDataset(encoded_lines, text_encoder.pad_idx, text_encoder.eos_idx, text_encoder.start_idx, train=train)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train,
                                collate_fn=dataset.collate_fn)

        return {
            'dataloader': dataloader,
            'encoder': text_encoder
        }

