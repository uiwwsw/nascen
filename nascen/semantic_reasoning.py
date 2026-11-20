"""Independent text-only student inference and an explicit rule-program control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .concept_model import ConceptModel, continuation_scores
from .sentence_encoder import MultilingualE5, SentenceVectors

if TYPE_CHECKING:
    from .semantic_teacher import PrincipleLesson

# A person supplies these condition checks from the training rubric.
# Each branch is a disjunction of conjunctions of literal cues.
RULE_PROGRAM = {
    "push": (
        (("저항", "큰 힘"), ("마찰", "이길 만큼"), ("마찰", "강하게"), ("저항", "넘는 힘")),
        (("저항", "작은 힘"), ("저항", "더 컸다"), ("마찰", "넘지 못"), ("마찰", "이기지 못"), ("버티는 힘", "작았다")),
    ),
    "latch": (
        (("걸쇠", "풀"), ("잠금쇠", "해제"), ("빗장", "빼"), ("빗장", "풀고")),
        (("걸쇠", "잠가"), ("걸쇠", "걸려"), ("잠금쇠", "맞물려"), ("빗장", "걸어")),
    ),
    "consent": (
        (("써도 된다고",), ("사용", "허락했다"), ("사용", "동의한다고"), ("빌려주겠다고",)),
        (("말을 한 적이 없다",), ("묻지 않았다",), ("허락한 적이 없다",), ("빌려준다고 하지 않았다",)),
    ),
    "scope": (
        (("골목에서만", "골목 안"), ("실내 촬영", "실내에서"), ("그림", "그리려"), ("집 안에서만", "집 안의")),
        (("골목에서만", "골목 밖"), ("실내 촬영", "야외"), ("그림", "게임"), ("집 안에서만", "길 건너")),
    ),
}


def context_vectors(context: tuple[str, ...], vectors: SentenceVectors) -> torch.Tensor:
    return torch.stack([vectors[sentence] for sentence in context])


@torch.no_grad()
def student_scores(student: ConceptModel, contexts: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
    """Inference consumes only vectors of raw sentences and candidate continuations."""
    return continuation_scores(student(contexts), options)


def _matches(text: str, alternatives) -> bool:
    return any(all(cue in text for cue in clause) for clause in alternatives)


def rule_choice(
    context: tuple[str, ...], options: torch.Tensor, lessons: dict[str, PrincipleLesson], vectors: SentenceVectors,
) -> int | None:
    """Run matched branches, then compare the resulting teacher prose to the candidates."""
    text = " ".join(context)
    margins = {}
    for key, lesson in lessons.items():
        yes, no = RULE_PROGRAM[key]
        if _matches(text, no):
            selected, opposite = lesson.otherwise_then, lesson.then
        elif _matches(text, yes):
            selected, opposite = lesson.then, lesson.otherwise_then
        else:
            continue
        margins[key] = options @ (vectors[selected] - vectors[opposite])

    # A consent exception overrides any scope branch. An out-of-scope use
    # requires new permission even when initial consent was given.
    if "consent" in margins and _matches(text, RULE_PROGRAM["consent"][1]):
        margins.pop("scope", None)
    elif "scope" in margins:
        margins.pop("consent", None)
    if not margins:
        return None
    scores = torch.stack(list(margins.values())).sum(dim=0)
    best = scores.max()
    return None if int((scores == best).sum()) != 1 else int(scores.argmax())


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank Korean continuations with an Experiment 007 student.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--context", nargs="+", required=True, help="ordered context sentences")
    parser.add_argument("--options", nargs="+", required=True, help="unlabelled candidate continuations")
    args = parser.parse_args()
    if len(set(args.options)) < 2:
        parser.error("provide at least two distinct continuations")
    torch.set_num_threads(1)
    try:
        encoder = MultilingualE5(args.encoder)
        checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
        if checkpoint["schema"] != 1 or checkpoint["encoder"] != encoder.identity:
            raise ValueError("Use the encoder snapshot from student training.")
        student = ConceptModel(encoder.width, checkpoint["seed"])
        student.load_state_dict(checkpoint["weights"])
        student.eval()
        vectors = SentenceVectors(encoder, [*args.context, *args.options])
        context = context_vectors(args.context, vectors).unsqueeze(0)
        options = context_vectors(args.options, vectors).unsqueeze(0)
        scores = student_scores(student, context, options).squeeze(0)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Semantic inference stopped: {error}\n")
    print(json.dumps({"choice": int(scores.argmax()), "cosine_scores": scores.tolist()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
