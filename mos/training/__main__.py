# from . import *
from mos.data.pipeline import load_data_pipeline
from mos.models.basic import Model
from mos.training.train import train_loop

def main():
    data = load_data_pipeline()

    model = Model(data['tokenizer'].num_tokens)
    train_loop(model, data)




if __name__ == '__main__':
    main()