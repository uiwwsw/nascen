"""Isolate outcome representation and slot regularization in the transition model."""

import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, cosine_similarity, cross_entropy, logsigmoid, normalize

from .model import TheoryLearner, rank

# A/B/C/D form a 2-by-2 comparison. E adds the existing reviewed theory signal to D.
VARIANTS = {
    "A": (False, False), "B": (True, False), "C": (False, True),
    "D": (True, True), "E": (True, True), "F": (True, False),
}


class RevisedLearner(TheoryLearner):
    def __init__(self, width, seed, variant):
        super().__init__(width, seed)
        self.project_outcomes, self.balance = VARIANTS[variant]
        self.change_gate = None
        if variant == "F":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 700)
                self.change_gate = nn.Sequential(nn.Linear(2 * self.hidden, self.hidden), nn.GELU(),
                                                 nn.Linear(self.hidden, 1))
        for kind in ("state", "action"):
            self.register_buffer(f"{kind}_center", torch.zeros(width))
            self.register_buffer(f"{kind}_scale", torch.ones(width))

    @torch.no_grad()
    def calibrate(self, data):
        if self.project_outcomes:
            for kind, examples in (
                ("state", torch.cat((data.before, data.equivalent_before, data.options.flatten(0, 1)))),
                ("action", torch.cat((data.action, data.equivalent_action))),
            ):
                getattr(self, f"{kind}_center").copy_(examples.mean(0))
                getattr(self, f"{kind}_scale").copy_(examples.std(0, unbiased=False).clamp_min(0.01))

    def _state(self, inputs):
        return self.state((inputs - self.state_center) / self.state_scale)

    def outcome_vectors(self, candidates):
        return normalize(self._state(candidates), dim=-1) if self.project_outcomes else candidates

    def step(self, state, action, *, uniform=False):
        encoded_action = self.action((action - self.action_center) / self.action_scale)
        context = torch.cat((state, encoded_action), dim=-1)
        logits = self.router(context)
        if self.balance:
            logits = 3 * logits.tanh()
        route = torch.full_like(logits, 1 / self.slots) if uniform else logits.softmax(dim=-1)
        mixture = route @ self.slot_values
        change = self.update(torch.cat((mixture, state * mixture), dim=-1))
        gate = self.change_gate(context).squeeze(-1) if self.change_gate is not None else None
        if gate is not None:
            change = change * gate.unsqueeze(-1).sigmoid()
        return self.norm(state + change), change, route, gate

    def forward(self, before, actions, *, uniform=False):
        state = self._state(before)
        ranking, reconstructions, changes, routes, priors, gates = [], [], [], [], [], []
        for action in actions.unbind(1):
            if self.change_gate is not None:
                priors.append(normalize(state, dim=-1))
            state, change, route, gate = self.step(state, action, uniform=uniform)
            if gate is not None:
                gates.append(gate)
            decoded = normalize(self.decode(state), dim=-1)
            ranking.append(normalize(state, dim=-1) if self.project_outcomes else decoded)
            reconstructions.append(decoded)
            changes.append(change)
            routes.append(route)
        return tuple(torch.stack(values, dim=1) if values else None
                     for values in (ranking, reconstructions, changes, routes, priors, gates))

    def outcome_scores(self, prediction, candidates, priors, gates):
        options = self.outcome_vectors(candidates)
        changed = rank(prediction, options) / 0.05
        if gates is None:
            return changed
        copied = (rank(priors, options) / 0.005).log_softmax(-1)
        return torch.logaddexp(changed.log_softmax(-1) + logsigmoid(gates.unsqueeze(-1)),
                               copied + logsigmoid(-gates.unsqueeze(-1)))


def _entropy(probabilities):
    return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)


def fit_revision(data, width, variant, seed, steps, batch_size, targets=None):
    model = RevisedLearner(width, seed, variant)
    model.calibrate(data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    generator = torch.Generator().manual_seed(seed + 300)
    for _ in range(steps):
        index = torch.randint(len(data.before), (min(len(data.before), batch_size),), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        prediction, decoded, change, routing, prior, gate = model(data.before[index], data.action[index, None])
        # Keep the fixed-encoder reconstruction loss in every arm.
        next_loss = 1 - cosine_similarity(decoded[:, 0], data.after[index]).mean()
        scores = model.outcome_scores(prediction, data.options[index, None], prior, gate)
        contrast = cross_entropy(scores[:, 0], torch.zeros(len(index), dtype=torch.long))
        alternate, _, _, alt_routing, _, _ = model(data.equivalent_before[index], data.equivalent_action[index, None])
        consistency = 1 - cosine_similarity(prediction[:, 0], alternate[:, 0]).mean()
        loss = next_loss + contrast + 0.2 * consistency
        if gate is not None:
            observed_change = (data.before[index] != data.after[index]).any(dim=-1).float()
            loss = loss + 0.5 * binary_cross_entropy_with_logits(gate[:, 0], observed_change)
        if model.balance:
            probabilities = routing[:, 0]
            # Prefer low-entropy individual routes with high-entropy aggregate usage.
            loss = loss + 0.25 * (_entropy(probabilities).mean() - _entropy(probabilities.mean(0)))
            loss = loss + 0.1 * (1 - cosine_similarity(routing, alt_routing).mean())
        if targets is not None:
            loss = loss + 0.3 * (1 - cosine_similarity(model.theory(change[:, 0]), targets[index]).mean())
        loss.backward()
        optimizer.step()
    return model.eval()
