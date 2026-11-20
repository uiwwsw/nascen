"""Hold cross-group principle combinations out of all case-based lessons."""

from random import Random

from .composition_world import STATES, Principle, Problem, trace

DEPTHS = (2, 3, 4, 5)
CASE_BUDGET = 1000
PAIRS_PER_DEPTH = 64


def crosses_groups(chain: tuple[int, ...]) -> bool:
    return any(index % 2 != chain[0] % 2 for index in chain[1:])


def training_problems(size: int, seed: int) -> list[Problem]:
    rng = Random(seed)
    order = rng.sample(range(size), size)
    atomic = min(200, 4 * size)
    problems = []
    seen = set()
    for index in range(CASE_BUDGET):
        depth = 1 if index < atomic else DEPTHS[(index - atomic) % len(DEPTHS)]
        for _ in range(5000):
            anchor = order[index % size] if index < max(size, atomic) else rng.randrange(size)
            same_group = [rule for rule in range(anchor % 2, size, 2) if rule != anchor]
            chain = rng.sample(same_group, depth - 1)
            chain.insert(rng.randrange(depth), anchor)
            problem = Problem(rng.randrange(len(STATES)), tuple(chain))
            if problem not in seen:
                break
        else:
            raise ValueError("Could not sample enough unique training problems.")
        problems.append(problem)
        seen.add(problem)
    return problems


def necessary(rules: list[Principle], problem: Problem, outcome: int) -> bool:
    return all(
        trace(rules, Problem(problem.state, problem.rules[:index] + problem.rules[index + 1:]))[-1] != outcome
        for index in range(len(problem.rules))
    )


def examination(rules: list[Principle], size: int, count: int = PAIRS_PER_DEPTH) -> list[tuple[Problem, Problem]]:
    rng = Random(606 + size)
    pairs = []
    seen = set()
    for depth in DEPTHS:
        start = len(pairs)
        for _ in range(count * 1000):
            chain = tuple(rng.sample(range(size), depth))
            if not crosses_groups(chain):
                continue
            change = list(chain)
            if (len(pairs) - start) % 2:
                first, second = rng.sample(range(depth), 2)
                change[first], change[second] = change[second], change[first]
            else:
                position = rng.randrange(depth)
                options = [index for index in range(chain[position] % 2, size, 2) if index not in chain]
                if not options:
                    continue
                change[position] = rng.choice(options)
            initial = rng.randrange(len(STATES))
            left, right = Problem(initial, chain), Problem(initial, tuple(change))
            if left in seen or right in seen:
                continue
            left_end, right_end = trace(rules, left)[-1], trace(rules, right)[-1]
            if left_end == right_end or not necessary(rules, left, left_end) or not necessary(rules, right, right_end):
                continue
            pairs.append((left, right))
            seen.update((left, right))
            if len(pairs) == start + count:
                break
        else:
            raise ValueError(f"Could not sample {count} decisive pairs at depth {depth}.")
    return pairs
