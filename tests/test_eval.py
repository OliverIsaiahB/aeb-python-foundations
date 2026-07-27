import pytest

from eval.models import Task, TaskKind, find_task
from eval.score import score_answer
from eval.metrics import summarize


def make_task(expected: str) -> Task:
    return Task(id="t1", kind=TaskKind.REASONING, prompt="q", expected=expected)


def test_score_exact_match_ignores_whitespace_and_case():
    assert score_answer(make_task("4"), "  4 ") == 1.0
    assert score_answer(make_task("4"), "Four") == 0.0


def test_summarize_empty_batch_is_all_zeros():
    s = summarize([])
    assert s.n == 0 and s.mean_score == 0.0 and s.pass_rate == 0.0


def test_summarize_mean_and_pass_rate():
    s = summarize([1.0, 0.0, 1.0, 1.0])
    assert s.n == 4
    assert s.mean_score == pytest.approx(0.75)
    assert s.pass_rate == pytest.approx(0.75)


def test_find_task_raises_on_missing():
    tasks = [make_task("4")]
    with pytest.raises(KeyError):
        find_task(tasks, "nope")
