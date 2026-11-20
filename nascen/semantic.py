"""Translate a local push observation and its measured effect into a reusable event."""

from dataclasses import dataclass

from .body import LocalView, Position
from .world import DIRECTIONS

PUSH_FIELDS = ("destination", "direction", *DIRECTIONS)
Offset = tuple[int, int]


@dataclass(frozen=True)
class PushObservation:
    position: Position
    direction: str
    neighbors: tuple[str, ...]

    @classmethod
    def from_view(cls, view: LocalView, target: int, direction: str) -> "PushObservation":
        position = view.objects[target].position
        return cls(position, direction, view.neighbors(position))

    @property
    def destination(self) -> str:
        return self.neighbors[tuple(DIRECTIONS).index(self.direction)]

    @property
    def features(self) -> tuple[str, ...]:
        return (self.destination, self.direction, *self.neighbors)

    def position_after(self, offset: Offset) -> Position:
        forward, sideways = offset
        dx, dy = DIRECTIONS[self.direction]
        return self.position[0] + forward * dx - sideways * dy, self.position[1] + forward * dy + sideways * dx


@dataclass(frozen=True)
class PushEvent:
    before: PushObservation
    after: Position

    @property
    def offset(self) -> Offset:
        dx, dy = DIRECTIONS[self.before.direction]
        x = self.after[0] - self.before.position[0]
        y = self.after[1] - self.before.position[1]
        return x * dx + y * dy, -x * dy + y * dx
