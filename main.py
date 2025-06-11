
# from DataProcessing import TextDataset, TextEncoder, TextExtractor, TextPipeline
# from torch.utils.data import DataLoader
# from Model import Seq2SeqModel, TrainPipeline, generate_text
# import torch

if __name__ == '__main__':
    # file_path = './facebook-firstnames.txt'
    file_path = './paul_graham_essay.txt'

    # train_data = TextPipeline.generate_dataset(file_path, True)
    # val_data = TextPipeline.generate_dataset(file_path, False)

    # model_path = './models/' + file_path[:-4] + '.pth'


    # vocab_size = train_data['encoder'].vocab_size

    # # model = torch.load(model_path, weights_only=False)

    # # generated_text = generate_text(model, train_data['encoder'], max_length=2000, temperature=0.5)
    # # print(f"Generated text: \"{generated_text}\"")
    # # generated_text = generate_text(model, train_data['encoder'], max_length=2000, temperature=1)
    # # print(f"Generated text: \"{generated_text}\"")

    # model = Seq2SeqModel(vocab_size)

    # TrainPipeline.train_loop(model, train_data['encoder'], train_data['dataloader'], val_data['dataloader'])

    # torch.save(model, model_path)
