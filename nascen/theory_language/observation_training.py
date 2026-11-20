"""Anchor sentence representations and learn the actual state transition from observations."""

from dataclasses import dataclass
from itertools import chain
from time import process_time

import torch
from torch.nn import functional as F

from .dataset import digest
from .encoder import encode_partition
from .observation_data import ACTION_CODES, STATE_CODES, clause_views, training_views
from .observation_encoding import encode_observations
from .observation_model import ObservationLearner


class EncodedViews:
    def __init__(self, groups, vectors):
        self.features = torch.stack([vectors[text] for texts in groups.values() for text in texts])
        counts = [len(group) for group in groups.values()]
        self.counts = torch.tensor(counts)
        self.offsets = self.counts.cumsum(0) - self.counts

    def draw(self, labels, generator):
        offsets = (torch.rand(labels.shape, generator=generator) * self.counts[labels]).long()
        return self.features[self.offsets[labels] + offsets]


@dataclass(frozen=True)
class EncodedObservations:
    states: EncodedViews
    actions: EncodedViews
    facts: EncodedViews
    transitions: torch.Tensor
    provenance: dict


def encode_training(encoder, windows=300, data_seed=42):
    states, actions, transitions = training_views(windows, data_seed)
    # A tiny smoke budget can omit rare state codes. Use observed state contexts
    # only, and tell the caller when a chosen real budget cannot calibrate a code.
    if states.keys() != set(range(len(STATE_CODES))):
        raise ValueError("The observation budget must cover all state codes. Increase --windows.")
    texts = chain.from_iterable((*states.values(), *actions.values()))
    vectors, cost = encode_observations(encoder, texts)
    facts = clause_views()
    fact_vectors, fact_cost = encode_partition(encoder, chain.from_iterable(facts.values()))
    provenance = {
        "windows": windows, "data_seed": data_seed, "transitions": len(transitions),
        "state_sentences": sum(map(len, states.values())), "action_sentences": sum(map(len, actions.values())),
        "clause_sentences": sum(map(len, facts.values())),
        "training_sha256": digest([states, actions, facts, transitions]), "encoding": cost, "clause_encoding": fact_cost,
    }
    return EncodedObservations(EncodedViews(states, vectors), EncodedViews(actions, vectors),
                               EncodedViews(facts, fact_vectors), torch.tensor(transitions), provenance)


def fit_observations(encoded, width, seed, steps=480, batch_size=64):
    if steps < 1 or batch_size < 1:
        raise ValueError("Use positive training step and batch counts.")
    state_views, action_views, transitions, facts = encoded.states, encoded.actions, encoded.transitions, encoded.facts
    model = ObservationLearner(width, len(STATE_CODES), len(ACTION_CODES), seed)
    model.calibrate(state_views.features, action_views.features, facts.features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    rng = torch.Generator().manual_seed(seed + 830)
    start = process_time()
    for _ in range(steps):
        offsets = torch.randint(len(transitions), (batch_size,), generator=rng)
        before, action, after = transitions[offsets].unbind(1)
        state_labels = torch.randint(model.states, (batch_size,), generator=rng)
        action_labels = torch.randint(model.actions, (batch_size,), generator=rng)
        fact_labels = torch.randint(len(facts.counts), (batch_size,), generator=rng)
        state_text_logits = model.state_logits(state_views.draw(state_labels, rng))
        action_text_logits = model.action_logits(action_views.draw(action_labels, rng))
        known_states = F.one_hot(before, num_classes=model.states).float()
        known_actions = F.one_hot(action, num_classes=model.actions).float()
        read_states = model.state_logits(state_views.draw(before, rng)).softmax(-1)
        read_actions = model.action_logits(action_views.draw(action, rng)).softmax(-1)
        loss = (
            F.cross_entropy(state_text_logits, state_labels)
            + F.cross_entropy(action_text_logits, action_labels)
            + F.cross_entropy(model.fact_logits(facts.draw(fact_labels, rng)), fact_labels)
            + F.cross_entropy(model.predict_state(known_states, known_actions), after)
            + F.cross_entropy(model.predict_state(read_states, read_actions), after)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval(), {
        "cpu_seconds": process_time() - start,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
