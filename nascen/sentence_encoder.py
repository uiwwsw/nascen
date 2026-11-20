"""Frozen Korean-capable sentence vectors and a surface-token comparison."""

from hashlib import blake2b, file_digest
from pathlib import Path
import re
import unicodedata

import torch
from torch.nn.functional import normalize

E5_MODEL = "intfloat/multilingual-e5-small"
E5_REVISION = "fd1525a9fd15316a2d503bf26ab031a61d056e98"
E5_FILES = (
    "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "sentencepiece.bpe.model",
)


class MultilingualE5:
    def __init__(self, snapshot: Path):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ValueError("Install the optional requirements-teacher.txt dependencies.") from error
        files = {}
        for name in E5_FILES:
            with (snapshot / name).open("rb") as stream:
                files[name] = file_digest(stream, "sha256").hexdigest()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True,
        )
        self.model = AutoModel.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        ).cpu().eval().requires_grad_(False)
        self.width = self.model.config.hidden_size
        self.input_tokens = 0
        self.identity = {
            "model": E5_MODEL, "reference_revision": E5_REVISION, "files_sha256": files,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "frozen": True, "pretraining_cost": None,
        }

    @torch.no_grad()
    def encode(self, sentences: tuple[str, ...]) -> torch.Tensor:
        vectors = []
        for start in range(0, len(sentences), 16):
            batch = ["query: " + sentence for sentence in sentences[start:start + 16]]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            self.input_tokens += int(inputs["attention_mask"].sum())
            tokens = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            vectors.append(normalize((tokens * mask).sum(1) / mask.sum(1), dim=-1))
        return torch.cat(vectors) if vectors else torch.empty(0, self.width)


class SurfaceTokens:
    def __init__(self, width: int = 384):
        self.width = width
        self.identity = {"model": "hashed Korean character and word n-grams", "parameters": 0, "pretraining_cost": 0}

    def encode(self, sentences: tuple[str, ...]) -> torch.Tensor:
        vectors = torch.zeros(len(sentences), self.width)
        for row, sentence in enumerate(sentences):
            words = re.findall(r"\w+", unicodedata.normalize("NFC", sentence).casefold())
            features = ["w:" + word for word in words]
            for word in words:
                marked = "^" + word + "$"
                features.extend(marked[i:i + size] for size in (2, 3) for i in range(len(marked) - size + 1))
            for token in features:
                value = int.from_bytes(blake2b(token.encode(), digest_size=4).digest())
                vectors[row, value % self.width] += 1 if value & (1 << 31) else -1
        return normalize(vectors, dim=-1)


class SentenceVectors:
    """Batch and reuse identical encodings across examples and trials."""

    def __init__(self, encoder, sentences):
        texts = tuple(dict.fromkeys(sentences))
        self.vectors = dict(zip(texts, encoder.encode(texts), strict=True))
        self.width = encoder.width

    def __getitem__(self, sentence: str) -> torch.Tensor:
        return self.vectors[sentence]
