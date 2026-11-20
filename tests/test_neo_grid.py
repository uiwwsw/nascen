"""Check that the frozen-program evaluation follows the upstream transfer protocol."""

import unittest

import torch

from nascen.neo_grid import (
    COMPOSITION_PROGRAMS, LONG_PROGRAMS, STATES, TRAIN_PROGRAMS, cells, episode_list,
    execute, fit, grid_split, induce, observations, transform,
)


class NEOGridTests(unittest.TestCase):
    def test_held_programs_and_query_states_are_disjoint_from_training(self):
        train, held = grid_split()
        observed_behavior = {
            tuple(transform(state, program) for state in STATES) for program in TRAIN_PROGRAMS
        }
        for program, support, after, query, outcome in episode_list(
            train, held, (*COMPOSITION_PROGRAMS, *LONG_PROGRAMS)
        ):
            self.assertNotIn(program, TRAIN_PROGRAMS)
            self.assertNotIn(tuple(transform(state, program) for state in STATES), observed_behavior)
            self.assertIn(support, train)
            self.assertIn(query, held)
            self.assertNotEqual(support[-1], query[-1])
            self.assertEqual(transform(query, program), outcome)
            self.assertNotEqual(support, after)

    def test_replay_matches_upstream_but_needs_only_the_query_start(self):
        train, held = grid_split()
        rows = episode_list(train, held, COMPOSITION_PROGRAMS)[:4]
        _, before, after, query, outcome = zip(*rows, strict=True)
        support_x, support_y, query_x, query_y = map(observations, (before, after, query, outcome))
        training_x = observations(train)
        training_y = observations([transform(state, TRAIN_PROGRAMS[0]) for state in train])
        model = fit(torch.stack((training_x, training_y), dim=1), seed=11, steps=2, max_length=2)

        with torch.no_grad():
            codes, lengths = induce(model, support_x, support_y, 2)
            actual = execute(model, query_x, codes, lengths)
            for targets in (query_y, query_y.roll(1, dims=0)):
                upstream_pairs = torch.stack((torch.stack((support_x, support_y), dim=1),
                                              torch.stack((query_x, targets), dim=1)), dim=1).flatten(0, 1)
                upstream = model(upstream_pairs, is_eval=True, hard_grounding=True, num_transitions=2)
                torch.testing.assert_close(upstream.selected_lengths[::2], lengths)
                torch.testing.assert_close(cells(upstream.predictions[1::2]), cells(actual))


if __name__ == "__main__":
    unittest.main()
