"""A text-grounded transition learner with a shared space for observed and predicted states."""

import torch
from torch import nn
from torch.nn import functional as F


class ObservationLearner(nn.Module):
    def __init__(self, width: int, states: int, actions: int, seed: int, hidden: int = 128):
        super().__init__()
        if states < 8 or states % 8:
            raise ValueError("Observation codes require eight states per domain.")
        self.width, self.states, self.actions, self.hidden = width, states, actions, hidden
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.state_encoder = nn.Sequential(nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                               nn.Linear(hidden, states))
            self.action_encoder = nn.Sequential(nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                                nn.Linear(hidden, actions))
            self.state_basis = nn.Linear(states, hidden, bias=False)
            self.action_basis = nn.Linear(actions, hidden, bias=False)
            self.transition = nn.Sequential(nn.Linear(hidden * 3, hidden * 2), nn.LayerNorm(hidden * 2),
                                            nn.GELU(), nn.Linear(hidden * 2, states))
            self.fact_encoder = nn.Sequential(nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                              nn.Linear(hidden, 6 * (states // 8) + 1))
        for kind in ("state", "action", "fact"):
            self.register_buffer(f"{kind}_center", torch.zeros(width))
            self.register_buffer(f"{kind}_scale", torch.ones(width))
        code = torch.arange(states)[:, None]
        bits = ((code % 8) >> torch.tensor([2, 1, 0])) & 1
        self.register_buffer("facts_per_state", 6 * (code // 8) + 2 * torch.arange(3) + bits)

    @torch.no_grad()
    def calibrate(self, states, actions, facts):
        banks = (("state", states[:, 0]), ("action", actions[:, 0]), ("fact", facts))
        for kind, inputs in banks:
            getattr(self, f"{kind}_center").copy_(inputs.mean(0))
            getattr(self, f"{kind}_scale").copy_(inputs.std(0, unbiased=False).clamp_min(0.01))

    def fact_logits(self, pieces):
        return self.fact_encoder((pieces - self.fact_center) / self.fact_scale)

    def state_logits(self, observations):
        whole = observations[..., 0, :]
        logits = self.state_encoder((whole - self.state_center) / self.state_scale)
        pieces = observations[..., 1:, :]
        evidence = F.log_softmax(self.fact_logits(pieces), dim=-1)
        evidence = evidence.masked_fill(~pieces.ne(0).any(dim=-1, keepdim=True), -1e4).amax(-2)
        properties = evidence[..., :-1].reshape(*evidence.shape[:-1], -1, 2).log_softmax(-1).flatten(-2)
        return 0.3 * logits + properties[..., self.facts_per_state].sum(-1)

    def action_logits(self, observations):
        return self.action_encoder((observations[..., 0, :] - self.action_center) / self.action_scale)

    def predict_state(self, state, action):
        state_features, action_features = self.state_basis(state), self.action_basis(action)
        context = torch.cat((state_features, action_features, state_features * action_features), dim=-1)
        return self.transition(context)

    def forward(self, before, actions):
        state = self.state_logits(before).softmax(-1)
        events = self.action_logits(actions).softmax(-1)
        predictions = []
        for action in events.unbind(1):
            prediction = self.predict_state(state, action)
            predictions.append(prediction)
            state = prediction.softmax(-1)
        return torch.stack(predictions, dim=1)

    def score_steps(self, before, actions, candidates):
        predicted = F.log_softmax(self(before, actions), dim=-1).unsqueeze(-2)
        meanings = F.log_softmax(self.state_logits(candidates), dim=-1)
        return (predicted + meanings).logsumexp(-1)


@torch.no_grad()
def predict_observations(model, encoded):
    predictions = {}
    for ids, before, actions, candidates in encoded:
        choices = model.score_steps(before, actions, candidates).argmax(-1).tolist()
        predictions.update(zip(ids, choices, strict=True))
    return predictions
