from .loader import load_text_file
from .tokenizer import load_tokenizer
from .dataset import get_text_dataloader

def load_data_pipeline():
    text_loader = load_text_file()
    print(f"Number of text lines processed: {len(text_loader.lines)}")

    text_tokenizer = load_tokenizer(text_loader.lines)
    print(f"Number of tokens: {text_tokenizer.num_tokens}")
    print(f"Length of encoded data: {len(text_tokenizer.encoded_data)}")

    train_dl, val_dl = get_text_dataloader(text_tokenizer)
    print(f"Train dataloader length: {len(train_dl)}")
    print(f"Val dataloader length: {len(val_dl)}")

    ret = {
        'dataloader': {'train': train_dl, 'val': val_dl},
        'tokenizer': text_tokenizer
    }
    return ret