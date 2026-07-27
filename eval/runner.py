import asyncio

from eval.models import Agent, Task
from eval.agent import run_agent_resilient
from eval.score import score_answer


async def run_one(agent: Agent, task: Task) -> float:
    """Run an agent on a task and score the result."""
    answer = await run_agent_resilient(agent, task)
    return score_answer(task, answer)


async def run_batch(agent: Agent, tasks: list[Task]) -> list[float]:
    """Run an agent over many tasks CONCURRENTLY and return all scores."""
    coros = [run_one(agent, task) for task in tasks]
    return await asyncio.gather(*coros)
