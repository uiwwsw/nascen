"""Learn to write principle text into a small, experience-updatable transition adapter."""

import torch
from torch import nn
from torch.nn import functional as F

HIDDEN = 32
FIELDS = 5


class PrincipleLearner(nn.Module):
    def __init__(self, width, seed):
        super().__init__()
        self.width = width
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.project = nn.Sequential(nn.Linear(width, HIDDEN), nn.LayerNorm(HIDDEN), nn.GELU())
            self.matcher = nn.Sequential(nn.Linear(4 * HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 2))
            self.transition = nn.Sequential(nn.Linear(3 + 2 * HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))
            self.prior = nn.Parameter(torch.randn(FIELDS, HIDDEN) * 0.05)
        self.register_buffer("center", torch.zeros(width))
        self.register_buffer("scale", torch.ones(width))

    @torch.no_grad()
    def calibrate(self, sentences):
        vectors = torch.cat([tensor.reshape(-1, self.width) for tensor in sentences])
        self.center.copy_(vectors.mean(0))
        self.scale.copy_(vectors.std(0, unbiased=False).clamp_min(0.025))

    def represent(self, inputs):
        return self.project((inputs - self.center) / self.scale)

    def write(self, lesson):
        return self.represent(lesson) if lesson is not None else self.prior

    def forward(self, facts, adapter):
        evidence = self.represent(facts)[:, None].expand(-1, 3, -1, -1)
        criteria = adapter[:, :3, None].expand_as(evidence)
        pairs = torch.cat((evidence, criteria, evidence * criteria, (evidence - criteria).abs()), dim=-1)
        relevance, truth = self.matcher(pairs).unbind(-1)
        satisfaction = (relevance.softmax(-1) * truth.sigmoid()).sum(-1)
        context = torch.cat((satisfaction, adapter[:, 3:].flatten(1)), dim=-1)
        return self.transition(context).squeeze(-1), relevance, truth


def fit_curriculum(data, width, seed, steps=480, batch_size=48):
    facts, lessons, observed, match, truth = data
    model = PrincipleLearner(width, seed)
    model.calibrate((facts, lessons))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    rng = torch.Generator().manual_seed(seed + 410)
    rows = torch.arange(batch_size)[:, None]
    criteria = torch.arange(3)[None, :]
    for _ in range(steps):
        index = torch.randint(len(facts), (batch_size,), generator=rng)
        adapter = model.write(lessons[index])
        logits, relevance, raw_truth = model(facts[index], adapter)
        baseline, _, _ = model(facts[index], model.prior.unsqueeze(0).expand_as(adapter))
        loss = (
            F.binary_cross_entropy_with_logits(logits, observed[index])
            + 0.25 * F.binary_cross_entropy_with_logits(baseline, observed[index])
            + F.cross_entropy(relevance.flatten(0, 1), match[index].flatten())
            + F.binary_cross_entropy_with_logits(raw_truth[rows, criteria, match[index]], truth[index])
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval().requires_grad_(False)


def study(model, lesson, support, labels, *, previous=None, steps=40):
    with torch.no_grad():
        initial = (model.write(lesson) if lesson is not None or previous is None else previous).clone()
    if support is None or len(support) == 0:
        return initial
    learned = nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([learned], lr=0.025)
    for _ in range(steps):
        logits = model(support, learned.unsqueeze(0).expand(len(support), -1, -1))[0]
        loss = F.binary_cross_entropy_with_logits(logits, labels) + 0.04 * (learned - initial).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return learned.detach()


@torch.no_grad()
def predict(model, adapter, facts):
    if adapter.shape != model.prior.shape:
        raise ValueError("The checkpoint principle weights do not match the model.")
    return model(facts, adapter.unsqueeze(0).expand(len(facts), -1, -1))[0].sigmoid()
