





class TextEncoder:
    def __init__(self, text_content) -> None:
        content = "".join(text_content)
        self.vocab_chars, self.vocab_size = self._get_vocab(content)
        self.char_to_idx, self.idx_to_char = self._get_encoding_and_decoding(self.vocab_chars)
        self.pad_idx = self.char_to_idx.get('<|pad|>', len(self.char_to_idx))
        self.eos_idx = self.char_to_idx.get('<|im_end|>', len(self.char_to_idx))
        self.start_idx = self.char_to_idx.get('<|im_start|>', len(self.char_to_idx))

    def _get_vocab(self, text_content):
        vocab_chars = sorted(list(set(text_content)) + ['<|pad|>', '<|im_end|>', '<|im_start|>'])
        return vocab_chars, len(vocab_chars)
        
    def _get_encoding_and_decoding(self, vocab_chars):
        char_to_idx = {c: i for i, c in enumerate(vocab_chars)}
        idx_to_char = {i: c for i, c in enumerate(vocab_chars)}
        return char_to_idx, idx_to_char
    
    def encode_lines(self, lines):
        return [self.encode(line) for line in lines]
    
    def encode(self, text_content):
        return [self.char_to_idx[char] for char in text_content]
    
    def decode(self, encoded_content):
        return "".join([self.idx_to_char[idx.item()] for idx in encoded_content])

class TiktokenTextEncoder:
    def __init__(self, text_content) -> None:
        cl100k_base = tiktoken.get_encoding("cl100k_base")
        self.encoder = tiktoken.Encoding(
            name="cl100k_im",
            pat_str=cl100k_base._pat_str,
            mergeable_ranks=cl100k_base._mergeable_ranks,
            special_tokens={
                **cl100k_base._special_tokens,
                "<|im_start|>": 100264,
                "<|im_end|>": 100265,
                "<|pad|>": 100266,
            }
        )
        # self.encoder = tiktoken.get_encoding("cl100k_base")
        self.start_idx = 100264
        self.eos_idx = 100265
        self.pad_idx = 100266
        self.vocab_size = self.encoder.n_vocab

    def encode_lines(self, lines):
        return [self.encode(line) for line in lines]
    
    def encode(self, text_content):
        return self.encoder.encode(text_content)
    
    def decode(self, encoded_content):
        return self.encoder.decode(encoded_content.tolist())
