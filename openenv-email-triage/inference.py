"""Baseline OpenAI inference loop for the Email Triage OpenEnv benchmark."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from openenv_email_triage.env.environment import EmailOpsEnvironment
from openenv_email_triage.env.tasks import TASKS
from openenv_email_triage.models.schemas import TriageAction, compact_action_string


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK_NAME = "email-triage-openenv"
MAX_MODEL_TURNS = 8

SYSTEM_PROMPT = """
You are operating a workplace email triage environment.
Choose exactly one next action at a time.
Available operations:
- read_email
- classify with value in [spam, customer_support, scheduling]
- route with value in [junk, billing, executive_ops]
- set_priority with value in [low, high, urgent]
- draft_reply with text
- submit

Rules:
- Return valid JSON only.
- Use keys: operation, value, text.
- Omit value when not needed.
- Omit text when not needed.
- Prefer incremental progress over guessing submit too early.
""".strip()

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def choose_action(observation: dict[str, Any], step_number: int) -> TriageAction:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "step": step_number,
                        "observation": observation,
                        "instruction": "Return the single best next action as JSON.",
                    },
                    ensure_ascii=True,
                ),
            },
        ],
    )
    payload = response.choices[0].message.content or "{}"
    return TriageAction(**json.loads(payload))


def run_task(task_id: str) -> None:
    env = EmailOpsEnvironment()
    rewards: list[str] = []
    steps = 0
    success = False
    print(f"[START] task={task_id} env={BENCHMARK_NAME} model={MODEL_NAME}")

    try:
        observation = env.reset(task_id=task_id)
        done = False
        while not done and steps < MAX_MODEL_TURNS:
            steps += 1
            action = choose_action(observation.model_dump(mode="json"), steps)
            next_observation, reward, done, info = env.step(action)
            rewards.append(f"{reward.delta:.2f}")
            error = info.get("last_action_error")
            error_text = "null" if error is None else str(error).replace("\n", " ")
            print(
                f"[STEP] step={steps} action={compact_action_string(action)} "
                f"reward={reward.delta:.2f} done={str(done).lower()} error={error_text}"
            )
            observation = next_observation
            success = done and info.get("score_fraction", 0.0) == 1.0
    except Exception as exc:
        error_text = str(exc).replace("\n", " ")
        print(
            f"[STEP] step={max(steps, 1)} action=exception() reward=0.00 done=true "
            f"error={error_text}"
        )
        success = False
    finally:
        env.close()
        rewards_text = ",".join(rewards) if rewards else "0.00"
        print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_text}")


if __name__ == "__main__":
    for task in TASKS:
        run_task(task.task_id)
