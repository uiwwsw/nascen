"""Compare single-step and unrolled training on identical simulated observations."""

import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import platform
from statistics import mean
from time import process_time

import torch
from torch.nn.functional import binary_cross_entropy_with_logits, cosine_similarity, cross_entropy

from .dataset import digest, write_json
from .encoder import MultilingualE5, encode_partition
from .evaluation import encode_questions, exam_texts, predict_copy
from .grounded_data import held_out_exam, observed_priors, structure_coverage, training_windows
from .revision import DEFAULT_ENCODER, candidate_only, predict_revision
from .revision_model import RevisedLearner
from .trainer import EncodedEpisodes, episode_texts

ARMS = {
    "single": (False, False),
    "single_grounded": (False, True),
    "rollout": (True, False),
    "rollout_grounded": (True, True),
}
WINDOW_LENGTH = 3


def training_inputs(data: EncodedEpisodes, window_ids: torch.Tensor, unroll: bool):
    indices = window_ids[:, None] * WINDOW_LENGTH + torch.arange(WINDOW_LENGTH)
    if not unroll:
        indices = indices.reshape(-1, 1)
    return indices, data.before[indices[:, 0]], data.action[indices]


def fit(data, width, arm, seed, steps, batch_size):
    unroll, ground = ARMS[arm]
    model = RevisedLearner(width, seed, "F")
    model.calibrate(data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    generator = torch.Generator().manual_seed(seed + 300)
    window_count = len(data.before) // WINDOW_LENGTH
    for _ in range(steps):
        window_ids = torch.randint(window_count, (min(batch_size, window_count),), generator=generator)
        index, before, actions = training_inputs(data, window_ids, unroll)
        optimizer.zero_grad(set_to_none=True)
        predicted, decoded, _, _, prior, gate = model(before, actions)
        scores = model.outcome_scores(predicted, data.options[index], prior, gate)
        contrast = cross_entropy(scores.flatten(0, 1), torch.zeros(index.numel(), dtype=torch.long))
        reconstruction = 1 - cosine_similarity(decoded, data.after[index], dim=-1).mean()
        alternate, *_ = model(data.equivalent_before[index[:, 0]], data.equivalent_action[index])
        consistency = 1 - cosine_similarity(predicted, alternate, dim=-1).mean()
        changes = (data.before[index] != data.after[index]).any(dim=-1).float()
        change_loss = binary_cross_entropy_with_logits(gate, changes)
        loss = contrast + reconstruction + 0.2 * consistency + 0.5 * change_loss
        if ground:
            observed = model.outcome_vectors(data.after[index]).detach()
            loss = loss + 0.5 * (1 - cosine_similarity(predicted, observed, dim=-1).mean())
        loss.backward()
        optimizer.step()
    return model.eval()


def score(keys, predictions):
    pair_groups = defaultdict(list)
    buckets = defaultdict(list)
    for key in keys:
        predicted = predictions[key.id]
        exact = tuple(p == answer for p, answer in zip(predicted, key.answers, strict=True))
        pair_groups[(key.skill, key.state_style, key.action_style, key.group)].append(all(exact))
        buckets[f"{key.skill}_exact"].append(all(exact))
        buckets[f"{key.skill}_final"].append(exact[-1])
        if key.skill == "order":
            buckets["later_steps"].extend(exact[1:])
            buckets[f"order_{key.world}_exact"].append(all(exact))
        previous = key.initial
        for step, (p, actual) in enumerate(zip(predicted, key.outcomes, strict=True)):
            guess = key.choices[step][p]
            for old, true, inferred in zip(previous, actual, guess, strict=True):
                kind = "changed" if old != true else "preserved"
                buckets[f"{key.skill}_{kind}_attribute"].append(inferred == true)
            previous = actual
    for (skill, state_style, action_style, _), answers in pair_groups.items():
        if len(answers) != 2:
            raise ValueError("A counterfactual group must have exactly two interventions.")
        buckets[f"{skill}_pair"].append(all(answers))
        state = "held_state" if state_style == 2 else "seen_state"
        action = "held_action" if action_style == 2 else "seen_action"
        buckets[f"{skill}_pair_{state}_{action}"].append(all(answers))
    return {key: mean(values) for key, values in sorted(buckets.items())}, {
        key: len(values) for key, values in sorted(buckets.items())
    }


def summarize(trials):
    rows = []
    for arm in dict.fromkeys(trial["arm"] for trial in trials):
        group = [trial for trial in trials if trial["arm"] == arm]
        metrics = group[0]["metrics"]
        row = {"arm": arm, "metrics": {name: mean(t["metrics"][name] for t in group) for name in metrics}}
        if "observed_previous_metrics" in group[0]:
            row["observed_previous_metrics"] = {
                name: mean(t["observed_previous_metrics"][name] for t in group) for name in metrics
            }
            row["seed_order_exact"] = [t["metrics"]["order_exact"] for t in group]
        rows.append(row)
    return rows


def run(encoder, windows=300, seeds=(11, 22, 42), steps=480, batch_size=16):
    windows = training_windows(windows)
    observations = tuple(row for window in windows for row in window)
    training_vectors, train_cost = encode_partition(encoder, episode_texts(observations))
    data = EncodedEpisodes.build(observations, training_vectors)
    questions, keys = held_out_exam()
    exam_vectors, exam_cost = encode_partition(encoder, exam_texts(questions))
    encoded = encode_questions(questions, exam_vectors)
    observed = encode_questions(observed_priors(questions, keys), exam_vectors)
    trials = []
    for name, guesses in (("candidate-only", candidate_only(encoded)), ("start-copy", predict_copy(encoded))):
        metrics, counts = score(keys, guesses)
        trials.append({"arm": name, "seed": None, "metrics": metrics, "predictions_sha256": digest(guesses)})
    examples = {}
    for arm in ARMS:
        for seed in seeds:
            start = process_time()
            model = fit(data, encoder.width, arm, seed, steps, batch_size)
            cpu_seconds = process_time() - start
            guesses, _ = predict_revision(model, encoded)
            one_step, _ = predict_revision(model, observed)
            privilege = {
                key.id: [one_step[f"{key.id}/{step}"][0] for step in range(len(key.answers))] for key in keys
            }
            trials.append({
                "arm": arm, "seed": seed, "training_cpu_seconds": cpu_seconds,
                "parameters": sum(p.numel() for p in model.parameters()),
                "metrics": score(keys, guesses)[0], "predictions_sha256": digest(guesses),
                "observed_previous_metrics": score(keys, privilege)[0],
                "observed_previous_predictions_sha256": digest(privilege),
            })
            if seed == seeds[0]:
                selected = {}
                for question, key in zip(questions, keys, strict=True):
                    category = (key.skill, key.state_style, key.action_style)
                    if category not in selected and tuple(guesses[key.id]) != key.answers:
                        selected[category] = asdict(question) | {
                            "answers": key.answers, "predicted": guesses[key.id],
                            "observed_previous_predicted": privilege[key.id],
                        }
                examples[arm] = list(selected.values())
    return {
        "experiment": "009 · Action effects and grounded rollouts",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "arch": platform.machine(),
                        "threads": torch.get_num_threads()},
        "windows": len(windows), "observations": len(observations),
        "unique_text_transitions": len({(row.before, row.action, row.after) for row in observations}),
        "update_steps": steps, "sampled_windows_per_step": min(batch_size, len(windows)), "seeds": seeds,
        "observations_per_update": min(batch_size, len(windows)) * WINDOW_LENGTH,
        "weights": {"outcome": 1, "reconstruction": 1, "paraphrase": 0.2, "change": 0.5, "grounding": 0.5},
        "train_sha256": digest([[asdict(row) for row in window] for window in windows]),
        "exam_sha256": digest([asdict(row) for row in questions]), "answer_sha256": digest([asdict(row) for row in keys]),
        "encoder": encoder.identity, "encoding_cost": {"train": train_cost, "exam": exam_cost},
        "metric_counts": counts,
        "structure_coverage": structure_coverage(windows, questions, keys),
        "conditions": {key: {"unrolled": unroll, "observed_state_loss": ground}
                       for key, (unroll, ground) in ARMS.items()},
        "summary": summarize(trials), "trials": trials, "errors_from_first_seed": examples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=300)
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--batch-size", type=int, default=16, help="length-three observation groups per update")
    parser.add_argument("--seeds", type=int, nargs="+", default=(11, 22, 42))
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use positive step and batch counts and distinct seeds.")
    torch.set_num_threads(1)
    try:
        result = run(MultilingualE5(args.encoder), args.windows, args.seeds, args.steps, args.batch_size)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Nascen grounded rollout: {error}\n")
    write_json(args.output, result)
    for row in result["summary"]:
        metrics = row["metrics"]
        print(f"{row['arm']} · action pair {metrics['one_step_pair']:.1%} · full rollout {metrics['order_exact']:.1%}")


if __name__ == "__main__":
    main()
