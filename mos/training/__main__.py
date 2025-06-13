import torch
from mos.data.pipeline import load_data_pipeline
from mos.models.basic import Model
from mos.training.train import train_loop

def main():
    data = load_data_pipeline()

    weights_dir = "./weights/abc.pth"

    print(data['tokenizer'].num_tokens)
    model = Model(data['tokenizer'].num_tokens)
    train_loop(model, data)

    torch.save(model, weights_dir)



if __name__ == '__main__':
    main()