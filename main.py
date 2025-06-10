
from DataProcessing import TextDataset, TextEncoder, TextExtractor, TextPipeline
from torch.utils.data import DataLoader
from Model import Seq2SeqModel, TrainPipeline


if __name__ == '__main__':
    file_path = './facebook-firstnames.txt'

    train_data = TextPipeline.generate_dataset(file_path, True)
    val_data = TextPipeline.generate_dataset(file_path, False)


    vocab_size = train_data['encoder'].vocab_size

    model = Seq2SeqModel(vocab_size)

    TrainPipeline.train_loop(model, train_data['encoder'], train_data['dataloader'], val_data['dataloader'])
