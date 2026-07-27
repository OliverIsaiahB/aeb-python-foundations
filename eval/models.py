from dataclasses import dataclass, field
from enum import Enum


class TaskKind(str, Enum):
    """The family a benchmark task belongs to — we score each kind differently."""
    REASONING = "reasoning"
    GAME = "game"
    SIMULATION = "simulation"


@dataclass(frozen=True)
class Task:
    """One benchmark task: a prompt plus the answer we grade against."""
    id: str
    kind: TaskKind
    prompt: str
    expected: str
    # Optional free-form notes (source, citation) — None when we have none.
    notes: str | None = None

    def label(self) -> str:
        """A short human label, e.g. 'reasoning:t1' — handy in logs."""
        return f"{self.kind.value}:{self.id}"


@dataclass(frozen=True)
class Agent:
    """An agent under test: a model plus the system prompt that steers it."""
    id: str
    model: str
    system_prompt: str
    temperature: float = 0.0
    tags: list[str] = field(default_factory=list)


def is_deterministic(agent: Agent) -> bool:
    """At temperature 0 the model is (near-)deterministic — repeatable runs."""
    return agent.temperature == 0.0


def find_task(tasks: list[Task], task_id: str) -> Task:
    """Return the task with this id, or raise a clear error if it's missing."""
    for task in tasks:
        if task.id == task_id:
            return task
    raise KeyError(f"no task with id {task_id!r}")
