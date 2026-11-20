"""Apply a learned motion principle inside the observation model's outcome scoring."""

import torch

from .observation_data import ACTION_CODES, STATE_CODES
from .principle_data import TRACK_DOMAIN
from .principle_model import predict
from .principle_study import load_bundle
from .principle_update import evidence_vectors

PHYSICAL_STATES = sum(world == "physical" for world, _ in STATE_CODES)
PUSH_INDEX = ACTION_CODES.index("push")


def apply_motion_probability(probabilities, moving):
    """Replace the motion marginal while preserving route and arrival marginals."""
    physical = probabilities[..., :PHYSICAL_STATES].unflatten(-1, (2, 4))
    retained = physical.sum(dim=-2, keepdim=True)
    motion = torch.stack((1 - moving, moving), dim=-1).unsqueeze(-1)
    updated = (motion * retained).flatten(-2)
    return torch.cat((updated, probabilities[..., PHYSICAL_STATES:]), dim=-1)


@torch.no_grad()
def predict_track(model, encoded, encoder, checkpoint, facts):
    principle, bundle = load_bundle(checkpoint, encoder)
    if bundle.get("domain") != TRACK_DOMAIN or bundle["adapter"] is None:
        raise ValueError("Use a track-motion checkpoint produced by principle study.")
    (ids, before, actions, options), = encoded
    if len(ids) != 1 or actions.shape[1] != 1:
        raise ValueError("Motion principles currently apply to one push at a time.")

    state = int(model.state_logits(before).argmax(-1).item())
    action = int(model.action_logits(actions).argmax(-1).item())
    if state >= PHYSICAL_STATES // 2 or action != PUSH_INDEX:
        raise ValueError("The motion principle applies to a push on a stationary object on the simulated track.")

    probability = predict(principle, bundle["adapter"], evidence_vectors(encoder, [facts]))
    logits = model(before, actions)
    base = logits.log_softmax(dim=-1)
    changed = apply_motion_probability(base.exp(), probability[:, None])
    baseline = int(model.score_candidates(base, options).argmax(-1).item())
    predicted = int(model.score_candidates(changed.log(), options).argmax(-1).item())
    return predicted, {"checkpoint": str(checkpoint), "baseline_index": baseline,
                       "moving_probability": float(probability[0])}
