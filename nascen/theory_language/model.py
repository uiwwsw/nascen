"""A latent transition model in which actions reach the state through learned slots."""

import torch
from torch import nn
from torch.nn.functional import normalize

HIDDEN = 128
SLOTS = 8


class TheoryLearner(nn.Module):
    def __init__(self, width: int, seed: int, hidden: int = HIDDEN, slots: int = SLOTS):
        super().__init__()
        self.width, self.hidden, self.slots = width, hidden, slots
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.state = nn.Sequential(nn.Linear(width, hidden), nn.LayerNorm(hidden))
            self.action = nn.Sequential(nn.Linear(width, hidden), nn.LayerNorm(hidden))
            self.router = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, slots))
            self.slot_values = nn.Parameter(torch.randn(slots, hidden) * 0.2)
            self.update = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            self.norm = nn.LayerNorm(hidden)
            self.decode = nn.Linear(hidden, width)
            self.theory = nn.Linear(hidden, width)

    def step(self, state, action, *, uniform=False):
        scores = self.router(torch.cat((state, self.action(action)), dim=-1))
        routing = torch.full_like(scores, 1 / self.slots) if uniform else scores.softmax(dim=-1)
        mixture = routing @ self.slot_values
        change = self.update(torch.cat((mixture, state * mixture), dim=-1))
        return self.norm(state + change), change, routing

    def forward(self, before, actions, *, uniform=False):
        state = self.state(before)
        predictions, changes, routes = [], [], []
        for action in actions.unbind(1):
            state, change, routing = self.step(state, action, uniform=uniform)
            predictions.append(normalize(self.decode(state), dim=-1))
            changes.append(change)
            routes.append(routing)
        return tuple(torch.stack(series, dim=1) for series in (predictions, changes, routes))


def rank(predictions, options):
    return (predictions.unsqueeze(-2) * options).sum(dim=-1)
