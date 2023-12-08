from __future__ import annotations


class Tokenizer:

    REMOVED_CHARS = [
        "\n",
        "\t",
        "\r",
        "\x0b",
        "\x0c",
        "\ufeff",
        "\u200b",
        "\u200e",
        "\u200f",
        "\u2028",
        "\u2029",
        "■",
        '□',
        '•',
        '~',
    ]

    def __init__(self) -> None:
        self.vocab_size = 0
        self.char_to_id = {}
        self.id_to_char = {}

    def _remove_chars(self, text: str) -> str:
        for c in self.REMOVED_CHARS:
            text = text.replace(c, "")
        return text
    def fit(self, text: str) -> None:
        chars = sorted(set(text.lower()))
        for c in self.REMOVED_CHARS:
            if c in chars:
                chars.remove(c)
        self.vocab_size = len(chars)
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        text = self._remove_chars(text)

        return [self.char_to_id[ch] for ch in text.lower()]

    def decode(self, ids: list[int]) -> str:
        return "".join([self.id_to_char[i] for i in ids])
    