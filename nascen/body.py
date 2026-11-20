"""Local sensing and movement in the embodied world."""

from dataclasses import dataclass

from .world import AGENT, DIRECTIONS, WALKS, Object, World, render_sprite

SIGHT_RADIUS = 2
Position = tuple[int, int]


@dataclass(frozen=True)
class VisibleObject:
    id: int
    position: Position
    pixels: tuple[float, ...]


@dataclass(frozen=True)
class LocalView:
    position: Position
    tiles: dict[Position, str]
    objects: dict[int, VisibleObject]

    def neighbors(self, center: Position) -> tuple[str, ...]:
        return tuple(self.tiles[shift(center, direction)] for direction in DIRECTIONS)


@dataclass(frozen=True)
class Action:
    name: str
    target: int | None = None


def shift(position: Position, direction: str) -> Position:
    dx, dy = DIRECTIONS[direction]
    return position[0] + dx, position[1] + dy


def distance(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


class EmbodiedWorld(World):
    def __init__(
        self, seed: int, scene: int, objects: list[Object] | None = None, position: Position = AGENT,
    ):
        super().__init__(seed, scene, objects)
        self.agent = position

    def look(self) -> LocalView:
        x, y = self.agent
        visible_tiles = {
            (column, row): self.tile((column, row))
            for column in range(x - SIGHT_RADIUS, x + SIGHT_RADIUS + 1)
            for row in range(y - SIGHT_RADIUS, y + SIGHT_RADIUS + 1)
            if distance(self.agent, (column, row)) <= SIGHT_RADIUS
        }
        visible_objects = {
            obj.id: VisibleObject(obj.id, obj.position, render_sprite(obj))
            for obj in self.objects.values()
            if obj.position in visible_tiles
        }
        return LocalView(self.agent, visible_tiles, visible_objects)

    @staticmethod
    def proposals(view: LocalView) -> list[Action]:
        actions = [Action(name) for name, direction in WALKS.items() if view.tiles[shift(view.position, direction)] == "empty"]
        for observed in view.objects.values():
            actions.extend((Action("inspect", observed.id), Action("ask", observed.id)))
            actions.extend(
                Action(direction, observed.id)
                for direction in DIRECTIONS
                if shift(view.position, direction) == observed.position
            )
        return actions or [Action("wait")]

    def allowed(self, obj: Object, action: str) -> bool:
        if action in DIRECTIONS:
            if shift(self.agent, action) != obj.position:
                return False
        elif distance(self.agent, obj.position) > SIGHT_RADIUS:
            return False
        return super().allowed(obj, action)

    def can_execute(self, action: Action) -> bool:
        if action.target is not None:
            obj = self.objects.get(action.target)
            return obj is not None and self.allowed(obj, action.name)
        if action.name in WALKS:
            return self.tile(shift(self.agent, WALKS[action.name])) == "empty"
        return action.name == "wait"

    def perform(self, action: Action) -> bool | None:
        """Execute a local action. Return the push outcome when there is one."""
        if not self.can_execute(action):
            raise ValueError("Action is outside reach or blocked by the world rules.")
        if action.name in WALKS:
            self.agent = shift(self.agent, WALKS[action.name])
        if action.target is not None:
            return self.execute(self.objects[action.target], action.name)
        return None
