"""Store every interaction and its run checkpoint in one SQLite transaction."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import json
import sqlite3

from .experiment import MEMORY_SIZE, STRATEGIES, Experiment

HISTORY_SIZE = 80
MAX_BATCH = 50
MAX_SEED = 2**31 - 1


class InvalidRequest(ValueError):
    pass


class StaleRun(InvalidRequest):
    pass


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY,
                    seed INTEGER NOT NULL,
                    strategy TEXT NOT NULL,
                    replay INTEGER NOT NULL,
                    checkpoint TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiences (
                    run_id INTEGER NOT NULL REFERENCES runs(id),
                    step INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (run_id, step)
                );
            """)
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT id FROM runs LIMIT 1").fetchone() is None:
                self._create(db, seed=42, strategy="active", replay=True)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _create(db: sqlite3.Connection, seed: int, strategy: str, replay: bool) -> tuple[int, Experiment]:
        experiment = Experiment(seed, strategy, replay)
        cursor = db.execute(
            "INSERT INTO runs (seed, strategy, replay, checkpoint) VALUES (?, ?, ?, ?)",
            (seed, strategy, replay, json.dumps(experiment.checkpoint())),
        )
        return cursor.lastrowid, experiment

    @staticmethod
    def _load(db: sqlite3.Connection, expected_run: int | None = None) -> tuple[int, Experiment]:
        run = db.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        if expected_run is not None and run["id"] != expected_run:
            raise StaleRun("This run changed. Refresh the lab and try again.")
        recent = db.execute(
            "SELECT payload FROM experiences WHERE run_id = ? ORDER BY step DESC LIMIT ?",
            (run["id"], MEMORY_SIZE),
        ).fetchall()
        experiment = Experiment(
            run["seed"], run["strategy"], bool(run["replay"]),
            state=json.loads(run["checkpoint"]),
            records=[json.loads(row["payload"]) for row in reversed(recent)],
        )
        return run["id"], experiment

    @staticmethod
    def _save(db: sqlite3.Connection, run_id: int, experiment: Experiment) -> None:
        db.execute(
            "UPDATE runs SET checkpoint = ? WHERE id = ?",
            (json.dumps(experiment.checkpoint()), run_id),
        )

    def get(self) -> dict:
        with self._connect() as db:
            db.execute("BEGIN")
            run_id, experiment = self._load(db)
            return public_state(run_id, experiment)

    def start(self, run_id: int, seed: int, strategy: str, replay: bool) -> dict:
        if type(seed) is not int or not 0 <= seed <= MAX_SEED:
            raise InvalidRequest(f"Seed must be an integer from 0 to {MAX_SEED}.")
        if strategy not in STRATEGIES or type(replay) is not bool:
            raise InvalidRequest("Choose active or random exploration and a replay setting.")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._load(db, run_id)
            new_id, experiment = self._create(db, seed, strategy, replay)
            return public_state(new_id, experiment)

    def step(self, run_id: int, count: int, teach_id: int | None = None, scene: int | None = None) -> dict:
        if type(count) is not int or not 1 <= count <= MAX_BATCH:
            raise InvalidRequest(f"Advance between 1 and {MAX_BATCH} steps.")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _, experiment = self._load(db, run_id)
            if teach_id is not None:
                if type(teach_id) is not int or teach_id not in experiment.world.objects or count != 1:
                    raise InvalidRequest("Choose an object for a single teaching interaction.")
                if scene != experiment.world.scene + 1:
                    raise StaleRun("The scene changed. Refresh the lab and choose an object again.")

            for _ in range(count):
                record = experiment.advance(teach_id)
                db.execute(
                    "INSERT INTO experiences (run_id, step, payload) VALUES (?, ?, ?)",
                    (run_id, record["step"], json.dumps(record)),
                )
            self._save(db, run_id, experiment)
            return public_state(run_id, experiment)

    def feedback(self, run_id: int, step: int, approval: int) -> dict:
        if type(approval) is not int or approval not in (-1, 1):
            raise InvalidRequest("Feedback must be +1 or −1.")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _, experiment = self._load(db, run_id)
            if not experiment.records or step != experiment.steps:
                raise StaleRun("The latest experience changed. Refresh before sending feedback.")
            if experiment.records[-1]["approval"] is not None:
                raise StaleRun("Feedback has already been recorded for this experience.")
            record = experiment.feedback(approval)
            db.execute(
                "UPDATE experiences SET payload = ? WHERE run_id = ? AND step = ?",
                (json.dumps(record), run_id, step),
            )
            self._save(db, run_id, experiment)
            return public_state(run_id, experiment)


def public_state(run_id: int, experiment: Experiment) -> dict:
    records = experiment.records
    latest = records[-1] if records else None
    episode = None
    if latest is not None:
        episode = {key: value for key, value in latest.items() if key not in ("visual", "context", "metrics", "target")}
        episode["label"] = (
            f'{latest["object"]["color"]} {latest["object"]["shape"]}' if latest["target"] is not None else None
        )
    history = [record["metrics"] for record in records[-HISTORY_SIZE:]]
    if experiment.steps < HISTORY_SIZE:
        history.insert(0, experiment.initial)
    return {
        "run_id": run_id, "seed": experiment.seed, "strategy": experiment.strategy, "replay": experiment.replay,
        "steps": experiment.steps, "questions": experiment.questions, "rejected": experiment.rejected,
        "world": experiment.public_world(), "latest": episode, "metrics": history[-1], "history": history,
    }
