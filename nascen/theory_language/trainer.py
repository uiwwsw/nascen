"""Train and save a teacher-independent TheoryLearner with explicit loss weights."""

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import process_time

import torch
from torch.nn.functional import cosine_similarity, cross_entropy, normalize

from .dataset import Episode
from .encoder import SentenceEncoder, SentenceVectors
from .model import TheoryLearner, rank

MODES = ("semantic", "principle", "shuffled", "noisy")
FALSE_PRINCIPLES = {
    "force": "길이 막혀 있어도 가만히 두면 물체가 자동으로 움직이기 시작한다.",
    "barrier": "이동 경로를 막아도 길이 열린 채로 남고 움직이던 물체의 운동도 그대로 유지된다.",
    "advance": "물체가 멈춰 있고 길이 막혀 있어도 시간이 지나면 목표선에 새로 도달한다.",
    "permission": "실제 소유자가 허락하지 않아도 관계없는 동료의 승인이 소유자의 승인으로 기록된다.",
    "time": "이미 오래전에 끝난 비상상황은 현재 비상상황으로 계속 처리한다.",
    "use": "범위를 벗어난 사용은 허락이나 비상상황의 여부와 상관없이 언제나 승인된다.",
}


@dataclass(frozen=True)
class LossWeights:
    next: float = 1
    contrast: float = 1
    principle: float = 0.3
    consistency: float = 0.2

    def __post_init__(self):
        if any(not isfinite(weight) or weight < 0 for weight in asdict(self).values()):
            raise ValueError("Every loss weight must be finite and nonnegative.")
        if self.next + self.contrast == 0:
            raise ValueError("Keep at least one outcome prediction loss.")


def episode_texts(episodes):
    for row in episodes:
        yield from (row.before, row.action, row.after, row.equivalent_before, row.equivalent_action, *row.negatives)


def lesson_texts(lessons):
    for lesson in lessons.values():
        yield from lesson.texts()
    yield from (FALSE_PRINCIPLES[key] for key in lessons)


@dataclass(frozen=True)
class EncodedEpisodes:
    before: torch.Tensor
    action: torch.Tensor
    after: torch.Tensor
    options: torch.Tensor
    equivalent_before: torch.Tensor
    equivalent_action: torch.Tensor

    @classmethod
    def build(cls, episodes: tuple[Episode, ...], vectors: SentenceVectors):
        return cls(
            torch.stack([vectors[row.before] for row in episodes]),
            torch.stack([vectors[row.action] for row in episodes]),
            torch.stack([vectors[row.after] for row in episodes]),
            torch.stack([torch.stack([vectors[text] for text in (row.after, *row.negatives)]) for row in episodes]),
            torch.stack([vectors[row.equivalent_before] for row in episodes]),
            torch.stack([vectors[row.equivalent_action] for row in episodes]),
        )


def theory_targets(episodes, lessons, vectors, mode, seed):
    if mode == "semantic":
        return None, {"annotated_episodes": 0, "incorrect_assignments": 0}
    prototypes = {
        key: normalize(torch.stack([vectors[text] for text in lesson.texts()]).mean(0), dim=0)
        for key, lesson in lessons.items()
    }
    annotated = [i for i, row in enumerate(episodes) if row.lesson in prototypes]
    if not annotated:
        raise ValueError("No episodes have reviewed principles.")
    keys = [row.lesson for row in episodes]
    assigned = list(keys)
    corrupted = set()
    rng = torch.Generator().manual_seed(seed + 1000)
    if mode == "shuffled":
        for index, donor in zip(annotated, torch.randperm(len(annotated), generator=rng).tolist(), strict=True):
            assigned[index] = keys[annotated[donor]]
    if mode == "noisy":
        selected = torch.randperm(len(annotated), generator=rng)[:len(annotated) // 4].tolist()
        corrupted.update(annotated[i] for i in selected)
    blank = torch.zeros(vectors.width)
    targets = torch.stack([
        vectors[FALSE_PRINCIPLES[key]] if i in corrupted else prototypes.get(key, blank)
        for i, key in enumerate(assigned)
    ])
    return targets, {
        "annotated_episodes": len(annotated),
        "incorrect_assignments": sum(keys[i] != assigned[i] or i in corrupted for i in annotated),
    }


def training_loss(model, data, index, targets, weights):
    before, actions = data.before[index], data.action[index, None]
    prediction, change, _ = model(before, actions)
    predicted = prediction[:, 0]
    next_loss = 1 - cosine_similarity(predicted, data.after[index]).mean()
    scores = rank(predicted, data.options[index])
    contrast = cross_entropy(scores / 0.05, torch.zeros(len(index), dtype=torch.long))
    consistency = predicted.new_zeros(())
    if weights.consistency:
        alt, _, _ = model(data.equivalent_before[index], data.equivalent_action[index, None])
        consistency = 1 - cosine_similarity(predicted, alt[:, 0]).mean()
    principle = predicted.new_zeros(())
    if targets is not None and weights.principle:
        desired = targets[index]
        present = (desired.norm(dim=-1) > 0).float()
        agreement = cosine_similarity(model.theory(change[:, 0]), desired)
        principle = ((1 - agreement) * present).sum() / present.sum().clamp_min(1)
    parts = {"next": next_loss, "contrast": contrast, "principle": principle, "consistency": consistency}
    total = sum(value * getattr(weights, key) for key, value in parts.items())
    return total, parts


def fit(data, width, seed, steps, batch_size, weights, targets=None):
    if steps < 1 or batch_size < 1 or len(data.before) < 1:
        raise ValueError("Training requires nonempty data and a positive step and batch count.")
    start = process_time()
    model = TheoryLearner(width, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    generator = torch.Generator().manual_seed(seed + 300)
    last = None
    for _ in range(steps):
        index = torch.randint(len(data.before), (min(len(data.before), batch_size),), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss, last = training_loss(model, data, index, targets, weights)
        loss.backward()
        optimizer.step()
    return model.eval(), {
        "cpu_seconds": process_time() - start,
        "last_batch_losses": {name: float(value.detach()) for name, value in last.items()},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def save_checkpoint(path: Path, model: TheoryLearner, encoder: SentenceEncoder, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save({
                "schema": 1, "encoder": encoder.identity, "width": model.width, "hidden": model.hidden,
                "slots": model.slots, "metadata": metadata, "weights": model.state_dict(),
            }, stream)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, encoder: SentenceEncoder):
    bundle = torch.load(path, weights_only=True, map_location="cpu")
    if bundle["schema"] != 1 or bundle["encoder"] != encoder.identity or bundle["width"] != encoder.width:
        raise ValueError("The student checkpoint requires its original encoder snapshot.")
    model = TheoryLearner(bundle["width"], 0, bundle["hidden"], bundle["slots"])
    model.load_state_dict(bundle["weights"])
    return model.eval(), bundle["metadata"]
