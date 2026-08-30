"""PaliGemma prompt tokenization used by pi0 and pi0.5."""

from __future__ import annotations

import logging
import pathlib

import numpy as np
import sentencepiece

LOGGER = logging.getLogger(__name__)


class PaligemmaTokenizer:
    def __init__(self, model_path: pathlib.Path, max_len: int = 48) -> None:
        path = model_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PaliGemma tokenizer model not found: {path}")
        self.max_len = max_len
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_proto=path.read_bytes())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            discretized = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            state_string = " ".join(map(str, discretized))
            text = f"Task: {cleaned}, State: {state_string};\nAction: "
            tokens = self.tokenizer.encode(text, add_bos=True)
        else:
            tokens = self.tokenizer.encode(cleaned, add_bos=True) + self.tokenizer.encode("\n")

        if len(tokens) > self.max_len:
            LOGGER.warning("Token length %d exceeds %d; truncating", len(tokens), self.max_len)
        tokens = tokens[: self.max_len]
        mask = [True] * len(tokens)
        padding = self.max_len - len(tokens)
        tokens += [0] * padding
        mask += [False] * padding
        return np.asarray(tokens, dtype=np.int64), np.asarray(mask, dtype=np.bool_)
