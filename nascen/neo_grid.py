"""Train KAIST's NEO on small before-and-after grids and test program transfer."""

import argparse
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import platform
from random import Random
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from ._vendor.neo import NEO
from .semantic_dataset import digest, write_json

UPSTREAM = "a3bf1bc7fcea498f3edc5dcc5b812b537a2b6a10"
SIZE = 3
FIELDS = 4
STATES = tuple(product(range(SIZE), repeat=FIELDS))
TRAIN_PROGRAMS = ((0,), (1,), (2,), (0, 2), (1, 2))
COMPOSITION_PROGRAMS = ((0, 1), (1, 0))
LONG_PROGRAMS = ((0, 1, 2), (1, 0, 2))


def transform(state, program):
    """Simulate hidden changes for the data generator and scoring only."""
    x, y, color, shape = state
    for operation in program:
        if operation == 0:
            x = (x + 1) % SIZE
        elif operation == 1:
            x, y = y, x
        elif operation == 2:
            color = (color + 1) % SIZE
        else:
            raise ValueError(f"Unknown grid operation {operation}.")
    return x, y, color, shape


def grid_split():
    train, held = [], []
    for state in STATES:
        x, y, color, shape = state
        (held if (x + 2 * y + color + shape) % SIZE == 2 else train).append(state)
    return train, held


def observations(states):
    return F.one_hot(torch.tensor(states), SIZE).float().flatten(-2).unsqueeze(-2)


def cells(values):
    return values.unflatten(-1, (FIELDS, SIZE)).argmax(-1).squeeze(-2)


class GridObjective:
    """Score exact equality of position, color and shape."""

    def validate_batch(self, batch, *, is_eval):
        if batch.ndim != 4 or batch.shape[1:] != (2, 1, FIELDS * SIZE):
            raise ValueError("Expected paired one-hot grid observations.")
        if is_eval and len(batch) % 2:
            raise ValueError("Evaluation uses matching support and query observations.")

    def split_batch(self, batch):
        return batch.unbind(dim=1)

    def reconstruction_loss(self, prediction, target):
        return F.mse_loss(prediction, target)

    def per_sample_reconstruction_loss(self, prediction, target):
        return (prediction - target).square().mean(dim=(1, 2))

    def exact_match(self, prediction, target):
        return cells(prediction).eq(cells(target)).all(dim=-1)

    def metrics(self, prediction, target):
        return self.exact_match(prediction, target).float().mean()

    def decode_prediction(self, prediction):
        return F.one_hot(cells(prediction), SIZE).to(prediction.dtype).flatten(-2).unsqueeze(-2)

    def empty_metrics(self):
        return torch.zeros(())


class FixedGridEncoder(nn.Module):
    """Expose the grid fields directly without learning a visual encoder."""

    def forward(self, values):
        return values, self

    def kl_to_standard_normal(self):
        return torch.zeros(())


@dataclass(frozen=True)
class Training:
    max_transition_length: int = 2
    state_dim: int = FIELDS * SIZE
    action_dim: int = 12
    num_state_tokens: int = 1
    num_action_tokens: int = 1
    policy_hidden_dim: int = 64
    transition_hidden_dim: int = 64
    dropout: float = 0.0
    action_codebook_size: int = 6
    action_commitment_weight: float = 0.25
    action_tau_start: float = 0.4
    action_tau_end: float = 0.05
    action_tau_steps: int = 500
    reconstruction_weight: float = 1.0
    grounding_weight: float = 0.08
    auto_reconstruction_weight: float = 0.0
    action_vq_weight: float = 0.1
    freeze_observation_encoder: bool = True
    freeze_observation_decoder: bool = True


def fit(pairs, seed, steps, max_length):
    torch.manual_seed(seed)
    config = Training()
    model = NEO(SimpleNamespace(training=config, length_control_coefficient=1.1),
                encoder=FixedGridEncoder(), decoder=nn.Identity(), objective=GridObjective())
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.003)
    indices = torch.Generator().manual_seed(seed + 200)
    for _ in range(steps):
        batch = pairs[torch.randint(len(pairs), (64,), generator=indices)]
        loss = model(batch, num_transitions=max_length).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.quantizer.step()
    return model.eval()


@torch.no_grad()
def induce(model, before, after, max_length):
    """Choose codes and stopping length using the support pair only."""
    state, _ = model.encoder(before)
    target, _ = model.encoder(after)
    codes, costs, matches = [], [], []
    for length in range(1, max_length + 1):
        action = model.theory_programmer(state, target)
        quantized = model.quantizer(action, training_mode=False)
        prediction = model.decoder(model.program_executor(state, quantized.values))
        state = model.objective.decode_prediction(prediction)
        costs.append(model.objective.per_sample_reconstruction_loss(prediction, after)
                     * model.length_control_coefficient ** length)
        matches.append(model.objective.exact_match(state, after))
        codes.append(quantized.indices)
    lengths = torch.stack(costs).argmin(dim=0) + 1
    for length, match in reversed(list(enumerate(matches, start=1))):
        lengths[match] = length
    return torch.stack(codes, dim=1), lengths


@torch.no_grad()
def execute(model, before, codes, lengths):
    """Replay frozen codes on a start state without consulting its outcome."""
    state, _ = model.encoder(before)
    for index, code in enumerate(codes.unbind(1)):
        action = model.quantizer.get_codebook_entry(code).unsqueeze(1)
        prediction = model.decoder(model.program_executor(state, action))
        grounded = model.objective.decode_prediction(prediction)
        state = torch.where((lengths > index)[:, None, None], grounded, state)
    return state


def episode_list(train, held, programs):
    rng = Random(410)
    rows = []
    for program in programs:
        for support in train:
            target = transform(support, program)
            if support == target:
                continue
            candidates = [query for query in held if query[-1] != support[-1]]
            for query in rng.sample(candidates, 2):
                rows.append((program, support, target, query, transform(query, program)))
    return rows


def measure(model, rows, depth):
    _, before, after, query, expected = zip(*rows, strict=True)
    support_x, support_y, query_x, query_y = (observations(states) for states in (before, after, query, expected))
    codes, lengths = induce(model, support_x, support_y, depth)
    support_ok = model.objective.exact_match(execute(model, support_x, codes, lengths), support_y)
    query_ok = model.objective.exact_match(execute(model, query_x, codes, lengths), query_y)
    by_program = {
        str(program): {
            "examples": sum(r[0] == program for r in rows),
            "support": sum(r[0] == program and bool(ok) for r, ok in zip(rows, support_ok, strict=True)),
            "transfer": sum(r[0] == program and bool(ok) for r, ok in zip(rows, query_ok, strict=True)),
        }
        for program in dict.fromkeys(row[0] for row in rows)
    }
    return {"support": int(support_ok.sum()), "transfer": int(query_ok.sum()),
            "both": int((support_ok & query_ok).sum()), "examples": len(rows), "by_program": by_program,
            "lengths": dict(sorted(Counter(lengths.tolist()).items())),
            "codes_used": sorted({code for row, length in zip(codes.tolist(), lengths.tolist(), strict=True)
                                  for code in row[:length]})}


def probe_single_operations(model, train, held):
    """Pick a single code on training starts and score its behavior on held starts."""
    count = model.experiment.training.action_codebook_size
    codes = torch.arange(count).repeat_interleave(len(train))[:, None]
    training_predictions = cells(execute(
        model, observations(train).repeat(count, 1, 1), codes, torch.ones(len(codes), dtype=torch.long)
    )).reshape(count, len(train), FIELDS)
    held_starts = observations(held)
    lengths = torch.ones(len(held), dtype=torch.long)
    probes = []
    for operation in range(3):
        training_targets = torch.tensor([transform(state, (operation,)) for state in train])
        matches = training_predictions.eq(training_targets).all(dim=-1).sum(dim=-1)
        best_code = int(matches.argmax())
        predicted = cells(execute(model, held_starts, torch.full((len(held), 1), best_code), lengths))
        targets = torch.tensor([transform(state, (operation,)) for state in held])
        probes.append({"operation": operation, "code": best_code, "train_correct": int(matches[best_code]),
                       "train_examples": len(train), "held_correct": int(predicted.eq(targets).all(dim=-1).sum()),
                       "held_examples": len(held)})
    return probes


def run(steps, seeds):
    train, held = grid_split()
    inputs, targets = zip(*[(state, transform(state, program)) for program in TRAIN_PROGRAMS for state in train],
                          strict=True)
    pairs = torch.stack((observations(inputs), observations(targets)), dim=1)
    exams = {name: episode_list(train, held, programs) for name, programs in
             (("seen_program", TRAIN_PROGRAMS), ("new_composition", COMPOSITION_PROGRAMS), ("longer", LONG_PROGRAMS))}
    trials = []
    for seed in seeds:
        for length in (1, 2):
            model = fit(pairs, seed, steps, length)
            scores = {name: measure(model, rows, length + int(name == "longer" and length > 1))
                      for name, rows in exams.items()}
            trials.append({"seed": seed, "training_max_length": length, "scores": scores,
                           "single_operation_probe": probe_single_operations(model, train, held)})
    return {"upstream_commit": UPSTREAM, "environment": {"python": platform.python_version(), "torch": torch.__version__,
                "machine": platform.machine(), "threads": torch.get_num_threads()},
            "steps": steps, "seeds": seeds, "training_programs": TRAIN_PROGRAMS,
            "held_programs": {"new_composition": COMPOSITION_PROGRAMS, "longer": LONG_PROGRAMS},
            "training_pairs": len(pairs), "train_start_states": len(train), "held_start_states": len(held),
            "exam_hashes": {name: digest(rows) for name, rows in exams.items()}, "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--seeds", type=int, nargs="+", default=(11, 22, 42))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Choose positive training steps and distinct seeds.")
    torch.set_num_threads(1)
    report = run(args.steps, args.seeds)
    write_json(args.output, report)
    for trial in report["trials"]:
        scores = " ".join(f"{name}={row['transfer']}/{row['examples']}" for name, row in trial["scores"].items())
        print(f"seed={trial['seed']} training_max_length={trial['training_max_length']} {scores}")


if __name__ == "__main__":
    main()
