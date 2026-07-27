import asyncio
import random

from openai import AsyncOpenAI

from eval.models import Agent, Task

# One async client, reused across all calls (it manages a connection pool).
client = AsyncOpenAI()


async def run_agent(agent: Agent, task: Task) -> str:
    """Run one agent on one task and return its raw answer text."""
    response = await client.chat.completions.create(
        model=agent.model,
        temperature=agent.temperature,
        messages=[
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": task.prompt},
        ],
    )
    return response.choices[0].message.content or ""


async def run_agent_resilient(
    agent: Agent, task: Task, *, attempts: int = 3, timeout_s: float = 30.0
) -> str:
    """Run an agent with a per-call timeout and retry transient failures."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return await asyncio.wait_for(run_agent(agent, task), timeout_s)
        except (asyncio.TimeoutError, ConnectionError) as exc:
            last_error = exc
            # Exponential backoff with jitter: 0.5s, 1s, 2s (± a little).
            delay = 0.5 * (2 ** attempt) + random.uniform(0, 0.25)
            await asyncio.sleep(delay)
    raise RuntimeError(f"run failed after {attempts} attempts") from last_error
