



class TextExtractor:
    file_path: str
    content: str
    content_length: int

    def __init__(self, file_path, max_content = 500000) -> None:
        self.file_path = file_path
        self.max_content = max_content
        self._load_contents_from_file()

    def _load_contents_from_file(self) -> None:
        with open(self.file_path, "r", encoding="UTF-8") as file:
            content = file.read()
            self.content = content[:self.max_content]

    def get_lines(self) -> list[str]:
        return self.content.strip().split('\n')
