from eval.models import Task, TaskKind


def score_exact(task: Task, answer: str) -> float:
    """Exact match after normalizing whitespace and case. Returns 1.0 or 0.0."""
    got = answer.strip().lower()
    want = task.expected.strip().lower()
    return 1.0 if got == want else 0.0


def score_contains(task: Task, answer: str) -> float:
    """Partial credit: 1.0 if the expected answer appears anywhere inside."""
    return 1.0 if task.expected.strip().lower() in answer.strip().lower() else 0.0


# Each task kind picks a scorer. Real platforms add LLM-judge and game scorers.
SCORERS = {
    TaskKind.REASONING: score_exact,
    TaskKind.GAME: score_exact,
    TaskKind.SIMULATION: score_contains,
}


def score_answer(task: Task, answer: str) -> float:
    """Dispatch to the scorer for this task's kind. The one entry point."""
    scorer = SCORERS[task.kind]
    return scorer(task, answer)
