import torch
import tiktoken
import string

from mos.utils.args import define_data_args

def load_tokenizer(text_data):
    args = define_data_args()
    tokenizer_type = args.tokenizer
    text_tokenizer = TextTokenizer(tokenizer_type, text_data).tokenizer
    return text_tokenizer

class TextTokenizer:
    def __init__(self, tokenizer_type, text_content):
        Tokenizer = CharacterTokenizer if tokenizer_type == 'char' else TiktokenTokenizer
        self.tokenizer = Tokenizer(text_content)

class CharacterTokenizer:
    def __init__(self, text_content):
        content = text_content if isinstance(text_content, str) else "".join(text_content)
        self._set_vocab_data(content)
        self.encoded_data = self.encode_lines(text_content)

    def _set_vocab_data(self, content):
        special_tokens = ['<|pad|>', '<|im_end|>', '<|im_start|>']

        tokens = self._get_tokens(content)
        self._set_token_encoding(tokens)
        self.special_tokens = self._add_special_tokens(special_tokens)
        self.num_tokens = len(self.char_to_idx)

    def _get_latin_characters(self):
        """
        Return all ASCII Latin alphabet characters for tokenization purposes.
        Includes uppercase A–Z and lowercase a–z.
        """
        return string.ascii_letters
    
    def _get_tokens(self, chars):
        vocab_chars = sorted(list(set(chars + self._get_latin_characters())))
        return vocab_chars
    
    def _set_token_encoding(self, tokens):
        self.char_to_idx = {c: i for i, c in enumerate(tokens)}
        self.idx_to_char = {i: c for i, c in enumerate(tokens)}

    def _add_special_tokens(self, special_tokens):
        special_token_info = {}
        for token in special_tokens:
            self.char_to_idx[token] = len(self.char_to_idx)
            self.idx_to_char[len(self.idx_to_char)] = token

            special_token_info[token] = self.char_to_idx[token]
        return special_token_info
    
    def encode(self, text_content: str):
        return torch.tensor([self.char_to_idx[char] for char in text_content], dtype=torch.long)
    
    def encode_lines(self, lines: list[str]):
        return [self.encode(line) for line in lines]
    
    def decode(self, tokens):
        return "".join([self.idx_to_char[tkn.item() if isinstance(tkn, torch.Tensor) else tkn] for tkn in tokens])

    def decode_lines(self, line_tokens):
        return "\n".join([self.decode(line) for line in line_tokens])

class TiktokenTokenizer:
    def __init__(self, text_content) -> None:
        special_tokens = ['<|pad|>', '<|im_end|>', '<|im_start|>']
        self.special_tokens = self._add_special_tokens(special_tokens)

        cl100k_base = tiktoken.get_encoding("cl100k_base")

        self.tokenizer = tiktoken.Encoding(
            name="cl100k_im",
            pat_str=cl100k_base._pat_str,
            mergeable_ranks=cl100k_base._mergeable_ranks,
            special_tokens=self.special_tokens
        )
        self.num_tokens = self.tokenizer.n_vocab
        self.encoded_data = self.encode_lines(text_content)

    def _add_special_tokens(self, special_tokens):
        cl100k_base = tiktoken.get_encoding("cl100k_base")
        special_token_base = {**cl100k_base._special_tokens}
        for token in special_tokens:
            special_token_base[token] = len(special_token_base)
        return special_token_base

    def encode_lines(self, lines):
        return [self.encode(line) for line in lines]
    
    def encode(self, text_content):
        return self.tokenizer.encode(text_content)
    
    def decode(self, encoded_content):
        return self.tokenizer.decode(encoded_content.tolist())