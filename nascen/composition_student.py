"""Train a shared neural transition model on end answers or individual principles."""

import torch
from torch import nn
from torch.nn.functional import cross_entropy, nll_loss, one_hot

from .composition_split import DEPTHS
from .composition_world import STATES, Problem

STATE_COUNT = len(STATES)
WIDTH = 32
EPOCHS = 240


class TransitionStudent(nn.Module):
    def __init__(self, size: int, seed: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.rules = nn.Embedding(size, WIDTH)
            self.decoder = nn.Sequential(nn.Linear(WIDTH, 64), nn.Tanh(), nn.Linear(64, STATE_COUNT ** 2))

    def forward(self, selected: torch.Tensor | None = None) -> torch.Tensor:
        embeddings = self.rules.weight if selected is None else self.rules(selected)
        return self.decoder(embeddings).reshape(-1, STATE_COUNT, STATE_COUNT)


def inputs(problems: list[Problem]) -> tuple[torch.Tensor, torch.Tensor]:
    initial = torch.tensor([problem.state for problem in problems])
    order = torch.tensor([list(problem.rules) + [-1] * (DEPTHS[-1] - len(problem.rules)) for problem in problems])
    return initial, order


def rollout(logits: torch.Tensor, initial: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    transitions = logits.softmax(dim=-1)
    state = one_hot(initial, STATE_COUNT).float()
    history = []
    for indices in order.unbind(dim=1):
        next_state = torch.bmm(state.unsqueeze(1), transitions[indices.clamp_min(0)]).squeeze(1)
        state = torch.where((indices >= 0).unsqueeze(1), next_state, state)
        history.append(state)
    return torch.stack(history, dim=1)


def fit(model: TransitionStudent, labels: torch.Tensor, problems: list[Problem] | None = None) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
    encoded = inputs(problems) if problems is not None else None
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        logits = model()
        if encoded is None:
            loss = cross_entropy(logits.flatten(0, 1), labels.flatten())
        else:
            final = rollout(logits, *encoded)[:, -1, :]
            loss = nll_loss(final.clamp_min(1e-9).log(), labels)
        loss.backward()
        optimizer.step()
    model.eval()


@torch.inference_mode()
def predict(model: TransitionStudent, problems: list[Problem]) -> list[tuple[int, ...]]:
    initial, order = inputs(problems)
    selected, local = order.clamp_min(0).unique(return_inverse=True)
    local = local.masked_fill(order < 0, -1)
    states = rollout(model(selected), initial, local).argmax(dim=-1).tolist()
    return [tuple(row[:len(problem.rules)]) for row, problem in zip(states, problems, strict=True)]


def decode_tables(tables: list[str]) -> list[list[int]]:
    return [[STATES.index(state) for state in table] for table in tables]


def execute(tables: list[list[int]], problems: list[Problem]) -> list[tuple[int, ...]]:
    predictions = []
    for problem in problems:
        state = problem.state
        history = []
        for index in problem.rules:
            state = tables[index][state]
            history.append(state)
        predictions.append(tuple(history))
    return predictions
