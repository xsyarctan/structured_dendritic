from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from transformers import AutoTokenizer


SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")


def _normalize(text: str, lowercase: bool) -> str:
    if text is None:
        return ""
    return text.lower() if lowercase else text


def basic_tokenize(text: str, mode: str, lowercase: bool = False) -> list[str]:
    text = _normalize(text, lowercase)
    if mode == "char":
        return list(text)
    if mode == "whitespace":
        return text.split()
    if mode == "wordpunct":
        return re.findall(r"\w+|[^\w\s]", text)
    raise ValueError(f"Unsupported tokenization mode: {mode}")


@dataclass
class TokenizerState:
    vocab_size: int
    pad_token_id: int
    bos_token_id: int | None
    eos_token_id: int | None


class VocabularyTokenizer:
    def __init__(
        self,
        mode: str,
        min_frequency: int = 1,
        lowercase: bool = False,
        add_bos: bool = False,
        add_eos: bool = True,
    ) -> None:
        self.mode = mode
        self.min_frequency = min_frequency
        self.lowercase = lowercase
        self.add_bos = add_bos
        self.add_eos = add_eos

        self.token_to_id = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = list(SPECIAL_TOKENS)

    def fit(self, texts: Iterable[str]) -> None:
        counter = Counter()
        for text in texts:
            counter.update(basic_tokenize(text, self.mode, self.lowercase))

        for token, count in counter.items():
            if count < self.min_frequency or token in self.token_to_id:
                continue
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        pieces = basic_tokenize(text, self.mode, self.lowercase)
        if max_length is not None:
            pieces = pieces[:max_length]

        ids = [self.token_to_id.get(piece, self.unk_token_id) for piece in pieces]
        if self.add_bos:
            ids = [self.bos_token_id] + ids
        if self.add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def state(self) -> TokenizerState:
        return TokenizerState(
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id if self.add_bos else None,
            eos_token_id=self.eos_token_id if self.add_eos else None,
        )


class HuggingFaceTokenizer:
    def __init__(self, name_or_path: str, add_bos: bool = False, add_eos: bool = True) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.add_bos = add_bos
        self.add_eos = add_eos

    def fit(self, texts: Iterable[str]) -> None:
        del texts

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        encoded = self.tokenizer(
            text or "",
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=True,
        )
        return encoded["input_ids"]

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def state(self) -> TokenizerState:
        return TokenizerState(
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )


def build_tokenizer(cfg, texts: Iterable[str]):
    if cfg.type == "huggingface":
        tokenizer = HuggingFaceTokenizer(
            name_or_path=cfg.name_or_path,
            add_bos=cfg.add_bos,
            add_eos=cfg.add_eos,
        )
    else:
        tokenizer = VocabularyTokenizer(
            mode=cfg.type,
            min_frequency=cfg.min_frequency,
            lowercase=cfg.lowercase,
            add_bos=cfg.add_bos,
            add_eos=cfg.add_eos,
        )

    tokenizer.fit(texts)
    return tokenizer
