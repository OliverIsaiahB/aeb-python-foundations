import asyncio

from eval.models import Agent, Task, TaskKind
from eval.runner import run_batch
from eval.metrics import summarize


def demo_agent() -> Agent:
    return Agent(
        id="a1",
        model="gpt-4o-mini",
        system_prompt="Answer with a single word. No punctuation.",
        tags=["baseline"],
    )


def demo_tasks() -> list[Task]:
    return [
        Task(id="t1", kind=TaskKind.REASONING, prompt="2 + 2 = ?", expected="4"),
        Task(id="t2", kind=TaskKind.REASONING, prompt="3 * 3 = ?", expected="9"),
        Task(id="t3", kind=TaskKind.REASONING, prompt="10 - 7 = ?", expected="3"),
    ]


async def main() -> None:
    agent = demo_agent()
    tasks = demo_tasks()
    scores = await run_batch(agent, tasks)
    summary = summarize(scores)
    print(
        f"agent={agent.id} n={summary.n} "
        f"mean={summary.mean_score:.2f} pass_rate={summary.pass_rate:.2f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
