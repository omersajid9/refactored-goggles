from .pipeline import load_data_pipeline

def main():
    train_dl, val_dl = load_data_pipeline()
    for batch_idx, (batch_x, batch_y, batch_len) in enumerate(train_dl):
        # batch_x dims => (batch_size, input_length)
        # batch_y dims => (batch_size, output_length)
        pass


if __name__ == "__main__":
    main()