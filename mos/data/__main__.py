from .pipeline import load_data_pipeline

def main():
    data = load_data_pipeline()
    for batch_idx, (batch_x, batch_y, batch_len) in enumerate(data['dataloader']['train']):

        print(f"Batch X [0]: {data['tokenizer'].decode(batch_x[0])}")
        print(f"Batch Y [0]: {data['tokenizer'].decode([batch_y[0]])}")
        # batch_x dims => (batch_size, input_length)
        # batch_y dims => (batch_size, output_length)

        # print(f)
        break


if __name__ == "__main__":
    main()