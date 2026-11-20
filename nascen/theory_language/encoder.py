"""Interface shared by the engine, benchmark and frozen E5 implementation."""

from time import process_time
from typing import Protocol

import torch

from ..sentence_encoder import MultilingualE5, SentenceVectors


class SentenceEncoder(Protocol):
    width: int
    identity: dict

    def encode(self, sentences: tuple[str, ...]) -> torch.Tensor: ...


def encode_partition(encoder: SentenceEncoder, sentences) -> tuple[SentenceVectors, dict]:
    start = process_time()
    initial_tokens = getattr(encoder, "input_tokens", None)
    vectors = SentenceVectors(encoder, sentences)
    return vectors, {
        "unique_sentences": len(vectors.vectors),
        "cpu_seconds": process_time() - start,
        "input_tokens": encoder.input_tokens - initial_tokens if initial_tokens is not None else None,
    }
