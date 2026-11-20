"""A bounded synthetic world and its simulator-owned teacher."""

from dataclasses import dataclass, replace
from random import Random

WIDTH, HEIGHT = 8, 6
MENTOR = (6, 2)
AGENT = (1, 4)
WALLS = ((1, 1), (5, 1), (3, 4), (4, 4), (6, 4))
SLOTS = ((2, 1), (2, 3), (5, 2), (6, 3))
COLORS = {
    "coral": (0.91, 0.33, 0.29),
    "amber": (0.91, 0.67, 0.23),
    "blue": (0.24, 0.51, 0.82),
    "violet": (0.64, 0.44, 0.79),
}
SHAPES = ("circle", "square", "triangle")
DIRECTIONS = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
ACTIONS = ("inspect", "ask", *DIRECTIONS)
PAIRINGS = tuple((color, shape) for color in COLORS for shape in SHAPES)
HELD_OUT = tuple((color, SHAPES[index % len(SHAPES)]) for index, color in enumerate(COLORS))
TRAIN_PAIRS = tuple(pair for pair in PAIRINGS if pair not in HELD_OUT)
TILES = ("empty", "wall", "object", "agent", "mentor")
SPRITE_SIZE = 7


@dataclass(frozen=True)
class Object:
    id: int
    color: str
    shape: str
    x: int
    y: int
    variant: int

    @property
    def position(self) -> tuple[int, int]:
        return self.x, self.y


class World:
    def __init__(self, seed: int, scene: int, objects: list[Object] | None = None):
        self.scene = scene
        if objects is None:
            rng = Random(seed * 100_003 + scene * 113 + 739)
            pairings = rng.sample(TRAIN_PAIRS, len(SLOTS))
            objects = [
                Object(index, color, shape, x, y, rng.randrange(1_000_000))
                for index, ((color, shape), (x, y)) in enumerate(zip(pairings, SLOTS, strict=True), 1)
            ]
        self.objects = {obj.id: obj for obj in objects}

    def tile(self, position: tuple[int, int]) -> str:
        x, y = position
        if not (0 <= x < WIDTH and 0 <= y < HEIGHT) or position in WALLS:
            return "wall"
        if position == MENTOR:
            return "mentor"
        if position == AGENT:
            return "agent"
        if any(obj.position == position for obj in self.objects.values()):
            return "object"
        return "empty"

    @staticmethod
    def destination(obj: Object, action: str) -> tuple[int, int]:
        dx, dy = DIRECTIONS[action]
        return obj.x + dx, obj.y + dy

    def neighbors(self, obj: Object) -> tuple[str, ...]:
        return tuple(self.tile(self.destination(obj, action)) for action in DIRECTIONS)

    def allowed(self, obj: Object, action: str) -> bool:
        return action in ACTIONS and (
            action not in DIRECTIONS or self.tile(self.destination(obj, action)) != "mentor"
        )

    def execute(self, obj: Object, action: str) -> bool | None:
        if not self.allowed(obj, action):
            raise ValueError("The mentor's tile is protected.")
        if action not in DIRECTIONS:
            return None
        destination = self.destination(obj, action)
        if self.tile(destination) != "empty":
            return False
        self.objects[obj.id] = replace(obj, x=destination[0], y=destination[1])
        return True


def observe(obj: Object) -> tuple[float, ...]:
    """Generate pixels, then expose a mask and average foreground RGB to the learner."""
    rng = Random(obj.variant)
    offset_x, offset_y = rng.choice((0, 0, -1, 1)), rng.choice((0, 0, -1, 1))
    scale = rng.choice((0.85, 0.95, 1.05))
    mask: list[float] = []
    channels: list[list[float]] = [[], [], []]
    for y in range(SPRITE_SIZE):
        for x in range(SPRITE_SIZE):
            px, py = (x - 3 - offset_x) / scale, (y - 3 - offset_y) / scale
            match obj.shape:
                case "circle":
                    visible = px * px + py * py <= 5
                case "square":
                    visible = abs(px) <= 2.25 and abs(py) <= 2.25
                case "triangle":
                    visible = -2.5 <= py <= 2.5 and abs(px) <= (py + 2.6) * 0.54
                case _:
                    raise ValueError(f"Unknown simulator shape {obj.shape!r}")
            mask.append(float(visible))
            if visible:
                for samples, base in zip(channels, COLORS[obj.color], strict=True):
                    samples.append(base + rng.uniform(-0.06, 0.06))
    rgb = tuple(sum(samples) / len(samples) for samples in channels)
    return (*rgb, *mask)


def concept_target(obj: Object) -> tuple[int, int]:
    return tuple(COLORS).index(obj.color), SHAPES.index(obj.shape)
