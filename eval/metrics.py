from dataclasses import dataclass


@dataclass(frozen=True)
class Summary:
    """Aggregate stats over a batch of scores — what a leaderboard row holds."""
    n: int
    mean_score: float
    pass_rate: float  # fraction of scores >= the pass threshold


def summarize(scores: list[float], pass_threshold: float = 1.0) -> Summary:
    """Reduce many scores to a single summary row. Empty batch -> all zeros."""
    n = len(scores)
    if n == 0:
        return Summary(n=0, mean_score=0.0, pass_rate=0.0)
    mean_score = sum(scores) / n
    n_passed = sum(1 for s in scores if s >= pass_threshold)
    return Summary(n=n, mean_score=mean_score, pass_rate=n_passed / n)
