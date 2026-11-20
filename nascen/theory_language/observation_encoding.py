"""Encode whole text and punctuation-delimited pieces without interpreting templates."""

from itertools import chain
import re

import torch

from .encoder import encode_partition
from .evaluation import encode_questions, exam_texts

MAX_PIECES = 7
_BOUNDARIES = re.compile(r"[.!?;,。！？；，\n]+")


def encode_observations(encoder, texts):
    pieces = {
        text: tuple(piece.strip() for piece in _BOUNDARIES.split(text) if piece.strip())[:MAX_PIECES] or (text,)
        for text in dict.fromkeys(texts)
    }
    vectors, cost = encode_partition(encoder, chain(pieces, chain.from_iterable(pieces.values())))
    padding = torch.zeros(encoder.width)
    observations = {
        text: torch.stack([vectors[text], *(vectors[part] for part in parts), *([padding] * (MAX_PIECES - len(parts)))])
        for text, parts in pieces.items()
    }
    return observations, cost


def encode_model_questions(encoder, questions, *, clauses=False):
    encode = encode_observations if clauses else encode_partition
    vectors, cost = encode(encoder, exam_texts(questions))
    return encode_questions(questions, vectors), cost
