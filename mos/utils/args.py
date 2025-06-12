import argparse


def define_data_args():
    parser = argparse.ArgumentParser(
                prog='mos',
                description='Data arguments')

    parser.add_argument('-tf', '--train-file', default="names")
    parser.add_argument('-ml', '--max-lines', default=1000)
    parser.add_argument('-tk', '--tokenizer', default="char")

    return get_args(parser)

def define_train_args():
    parser = argparse.ArgumentParser(
                prog='mos',
                description='Training arguments')

    parser.add_argument('-e', '--epochs')
    parser.add_argument('-lr', '--learning-rate')
    parser.add_argument('-do', '--dropout')

    return get_args(parser)


def get_args(parser):
    return parser.parse_args()