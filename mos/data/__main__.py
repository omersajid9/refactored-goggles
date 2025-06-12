from .loader import load_text_file
from .encoder import TextTokenizer, load_tokenizer

def main():
    text_loader = load_text_file()
    print(f"Number of text lines processed: {len(text_loader.lines)}")

    text_tokenizer = load_tokenizer(text_loader.lines)
    print(f"Number of tokens: {text_tokenizer.num_tokens}")
    print(f"Length of encoded data: {len(text_tokenizer.encoded_data)}")

    


        #     dataset = TextDataset(encoded_lines, text_encoder.pad_idx, text_encoder.eos_idx, text_encoder.start_idx, train=train)

        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train,
        #                         collate_fn=dataset.collate_fn)

    
    pass

if __name__ == "__main__":
    main()