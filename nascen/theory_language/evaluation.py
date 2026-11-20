"""Predict with raw text and score with the separate answer key."""

from collections import defaultdict
from statistics import mean
from time import process_time

import torch

from .model import rank


def exam_texts(questions):
    for row in questions:
        yield row.before
        yield from row.actions
        for options in row.options:
            yield from options


def encode_questions(questions, vectors):
    grouped = defaultdict(list)
    for row in questions:
        if not row.actions or len(row.actions) != len(row.options):
            raise ValueError("Supply candidate outcomes for each action.")
        option_counts = {len(options) for options in row.options}
        if len(option_counts) != 1 or min(option_counts) < 2:
            raise ValueError("Supply the same number of at least two candidate outcomes for each action.")
        grouped[len(row.actions), option_counts.pop()].append(row)
    return [
        (
            [row.id for row in rows],
            torch.stack([vectors[row.before] for row in rows]),
            torch.stack([torch.stack([vectors[text] for text in row.actions]) for row in rows]),
            torch.stack([torch.stack([torch.stack([vectors[text] for text in options])
                                     for options in row.options]) for row in rows]),
        )
        for rows in grouped.values()
    ]


@torch.no_grad()
def predict(model, encoded, *, uniform=False):
    predictions = {}
    routes = []
    start = process_time()
    for ids, before, actions, candidates in encoded:
        predicted, _, routing = model(before, actions, uniform=uniform)
        choices = rank(predicted, candidates).argmax(dim=-1).tolist()
        predictions.update(zip(ids, choices, strict=True))
        routes.append(routing.flatten(0, 1))
    return predictions, torch.cat(routes), process_time() - start


def predict_rules(engine, encoded):
    predictions = {}
    for ids, before, actions, options in encoded:
        predictions.update(
            (identifier, engine.predict(state, events, choices))
            for identifier, state, events, choices in zip(ids, before, actions, options, strict=True)
        )
    return predictions


def predict_copy(encoded):
    return {
        identifier: choices for ids, before, _, options in encoded
        for identifier, choices in zip(ids, rank(before[:, None, :], options).argmax(dim=-1).tolist(), strict=True)
    }


def route_statistics(routes):
    marginal = routes.mean(0)
    entropy = -(marginal * marginal.clamp_min(1e-8).log()).sum()
    return {
        "effective_slots": float(entropy.exp()),
        "mean_slot_mass": marginal.tolist(),
        "max_slot_mass": float(marginal.max()),
        "top_slot_counts": torch.bincount(routes.argmax(-1), minlength=len(marginal)).tolist(),
    }


def grade(answers, predictions):
    if predictions.keys() != {row.id for row in answers}:
        raise ValueError("Predictions and exam answer IDs do not match.")
    by_skill = defaultdict(list)
    pairs = defaultdict(dict)
    in_domain = []
    whole_chain, intermediate, final_chain = [], [], []
    for row in answers:
        predicted = predictions[row.id]
        checks = [actual == expected for actual, expected in zip(predicted, row.steps, strict=True)]
        by_skill[row.skill].append(checks[-1])
        pairs[row.skill].setdefault(row.group, []).append(checks[-1])
        if row.skill != "outside_knowledge":
            in_domain.append(checks[-1])
        if row.skill == "composition":
            whole_chain.append(all(checks))
            intermediate.extend(checks[:-1])
            final_chain.append(checks[-1])
    return {
        "accuracy": mean(in_domain),
        "by_skill": {key: {"questions": len(values), "accuracy": mean(values)} for key, values in by_skill.items()},
        "both_correct": {
            skill: mean(all(group) for group in values.values())
            for skill, values in pairs.items() if all(len(group) == 2 for group in values.values())
        },
        "composition_intermediate": mean(intermediate),
        "composition_final": mean(final_chain),
        "composition_all_steps": mean(whole_chain),
    }
