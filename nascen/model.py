"""Small CPU networks trained directly from simulated interactions."""

from collections.abc import Sequence

import torch
from torch import nn

from .world import ACTIONS, COLORS, DIRECTIONS, SHAPES, SPRITE_SIZE, TILES

VISUAL_SIZE = 3 + SPRITE_SIZE**2
CONTEXT_SIZE = len(DIRECTIONS) * len(TILES)


class Learner(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        # Local initialization keeps independent runs reproducible even within a shared process.
        generator = torch.Generator().manual_seed(seed)
        self.color = nn.Linear(3, len(COLORS))
        self.shape = nn.Sequential(nn.Linear(SPRITE_SIZE**2, 16), nn.Tanh(), nn.Linear(16, len(SHAPES)))
        self.physics = nn.Sequential(nn.Linear(CONTEXT_SIZE + len(DIRECTIONS), 16), nn.Tanh(), nn.Linear(16, 1))
        self.preference = nn.Linear(VISUAL_SIZE + CONTEXT_SIZE, len(ACTIONS))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5, generator=generator)
                nn.init.zeros_(layer.bias)

    def concepts(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.color(inputs[..., :3]), self.shape(inputs[..., 3:])

    def motion(self, context: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        encoded = nn.functional.one_hot(directions, len(DIRECTIONS)).float()
        return self.physics(torch.cat((context, encoded), dim=-1)).squeeze(-1)

    def rewards(self, visual: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.preference(torch.cat((visual, context), dim=-1))

    def checkpoint(self) -> dict:
        return {name: weight.tolist() for name, weight in self.state_dict().items()}

    def restore(self, weights: dict) -> None:
        self.load_state_dict({name: torch.tensor(value) for name, value in weights.items()})

    def train_concepts(self, labeled: Sequence[tuple[list[float], list[int]]]) -> None:
        inputs = torch.tensor([features for features, _ in labeled])
        targets = torch.tensor([target for _, target in labeled])
        colors, shapes = self.concepts(inputs)
        loss = nn.functional.cross_entropy(colors, targets[:, 0]) + nn.functional.cross_entropy(shapes, targets[:, 1])
        _learn(loss, [*self.color.parameters(), *self.shape.parameters()], rate=0.32)

    def train_motion(self, transitions: Sequence[tuple[list[float], int, bool]]) -> None:
        context = torch.tensor([features for features, _, _ in transitions])
        actions = torch.tensor([direction for _, direction, _ in transitions])
        actual = torch.tensor([float(moved) for _, _, moved in transitions])
        loss = nn.functional.binary_cross_entropy_with_logits(self.motion(context, actions), actual)
        _learn(loss, self.physics.parameters(), rate=0.23)

    def train_reward(self, visual: list, context: list, action: str, reward: float) -> None:
        predicted = self.rewards(torch.tensor(visual), torch.tensor(context))[ACTIONS.index(action)]
        loss = nn.functional.smooth_l1_loss(predicted, predicted.new_tensor(reward))
        _learn(loss, self.preference.parameters(), rate=0.025)


def context_features(neighbors: tuple[str, ...]) -> list[float]:
    return [float(tile == candidate) for tile in neighbors for candidate in TILES]


def _learn(loss: torch.Tensor, parameters, rate: float) -> None:
    parameters = tuple(parameters)
    gradients = torch.autograd.grad(loss, parameters)
    with torch.no_grad():
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter.add_(gradient, alpha=-rate)
