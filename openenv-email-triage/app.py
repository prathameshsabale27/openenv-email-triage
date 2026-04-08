"""Gradio UI for the email triage environment."""

from __future__ import annotations

import json
import os

import gradio as gr
from openai import OpenAI

from openenv_email_triage.env.environment import EmailOpsEnvironment
from openenv_email_triage.env.tasks import TASKS
from openenv_email_triage.models.schemas import TriageAction


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

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

Return valid JSON with keys: operation, value, text.
Omit value or text when they are not needed.
""".strip()


def _new_session(task_id: str) -> dict:
    env = EmailOpsEnvironment()
    observation = env.reset(task_id=task_id)
    return {
        "task_id": task_id,
        "observation": observation.model_dump(mode="json"),
        "actions": [],
        "history": [],
    }


def _restore_env(session: dict) -> EmailOpsEnvironment:
    env = EmailOpsEnvironment()
    env.reset(task_id=session["task_id"])
    for payload in session["actions"]:
        env.step(payload)
    return env


def _render_observation(session: dict) -> tuple[str, str, str, str, str, str]:
    observation = session["observation"]
    email = observation["email"]
    workspace = observation["workspace"]
    history = session["history"]

    email_md = (
        f"**From:** {email['sender']}\n\n"
        f"**Subject:** {email['subject']}\n\n"
        f"{email['body']}"
    )
    workspace_md = (
        f"- classification: `{workspace['classification']}`\n"
        f"- queue: `{workspace['queue']}`\n"
        f"- priority: `{workspace['priority']}`\n"
        f"- draft_reply: `{workspace['draft_reply']}`"
    )
    requirements_md = (
        f"Completed: `{', '.join(observation['completed_requirements']) or 'none'}`\n\n"
        f"Pending: `{', '.join(observation['pending_requirements']) or 'none'}`"
    )
    status_md = (
        f"Task: `{observation['task_id']}`\n\n"
        f"Difficulty: `{observation['difficulty']}`\n\n"
        f"Steps remaining: `{observation['steps_remaining']}`\n\n"
        f"Last error: `{observation['last_action_error']}`\n\n"
        f"Reward: `{observation['reward']}`\n\n"
        f"Done: `{observation['done']}`"
    )
    history_md = "\n".join(history) if history else "No actions taken yet."
    return (
        observation["objective"],
        email_md,
        workspace_md,
        requirements_md,
        status_md,
        history_md,
    )


def reset_task(task_id: str) -> tuple[str, str, str, str, str, str, dict]:
    session = _new_session(task_id)
    objective, email_md, workspace_md, requirements_md, status_md, history_md = _render_observation(
        session
    )
    return objective, email_md, workspace_md, requirements_md, status_md, history_md, session


def apply_action(
    operation: str,
    value: str,
    text: str,
    session: dict | None,
) -> tuple[str, str, str, str, str, str, dict]:
    if not session:
        session = _new_session(TASKS[0].task_id)

    action_payload = {"operation": operation}
    if value.strip():
        action_payload["value"] = value.strip()
    if text.strip():
        action_payload["text"] = text.strip()

    env = _restore_env(session)
    action = TriageAction(**action_payload)
    observation, reward, done, info = env.step(action)

    session["actions"].append(action.model_dump(exclude_none=True))
    session["observation"] = observation.model_dump(mode="json")
    session["history"].append(f"ACTION {json.dumps(action.model_dump(exclude_none=True))}")
    session["history"].append(
        f"RESULT reward={reward.delta:.2f} progress={info['score_fraction']:.2f} done={str(done).lower()} error={info['last_action_error']}"
    )

    objective, email_md, workspace_md, requirements_md, status_md, history_md = _render_observation(
        session
    )
    return objective, email_md, workspace_md, requirements_md, status_md, history_md, session


def suggest_action(session: dict | None) -> str:
    if not session:
        return "Initialize a task first."
    if HF_TOKEN is None:
        return "HF_TOKEN is not set, so AI suggestions are unavailable."

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
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
                        "instruction": "Return the single best next action as JSON.",
                        "observation": session["observation"],
                    },
                    ensure_ascii=True,
                ),
            },
        ],
    )
    return response.choices[0].message.content or "{}"


with gr.Blocks(title="Email Triage OpenEnv") as demo:
    session_state = gr.State()

    gr.Markdown("# Email Triage OpenEnv")
    gr.Markdown(
        "A Gradio front end for the real OpenEnv email-triage tasks. You can step through the same environment used by the baseline inference script and server runtime."
    )

    with gr.Row():
        task_picker = gr.Dropdown(
            choices=[task.task_id for task in TASKS],
            value=TASKS[0].task_id,
            label="Task",
        )
        reset_button = gr.Button("Reset Task")

    objective_box = gr.Markdown()
    with gr.Row():
        email_box = gr.Markdown()
        status_box = gr.Markdown()
    with gr.Row():
        workspace_box = gr.Markdown()
        requirements_box = gr.Markdown()

    gr.Markdown("## Take Action")
    operation_box = gr.Dropdown(
        choices=["read_email", "classify", "route", "set_priority", "draft_reply", "submit"],
        value="read_email",
        label="Operation",
    )
    value_box = gr.Textbox(label="Value", placeholder="spam, billing, urgent, ...")
    text_box = gr.Textbox(label="Reply Draft", lines=4)
    step_button = gr.Button("Apply Action")

    gr.Markdown("## History")
    history_box = gr.Markdown()

    gr.Markdown("## AI Suggestion")
    suggestion_box = gr.Code(label="Suggested JSON", language="json")
    suggest_button = gr.Button("Suggest Next Action")

    reset_button.click(
        reset_task,
        inputs=task_picker,
        outputs=[
            objective_box,
            email_box,
            workspace_box,
            requirements_box,
            status_box,
            history_box,
            session_state,
        ],
    )
    step_button.click(
        apply_action,
        inputs=[operation_box, value_box, text_box, session_state],
        outputs=[
            objective_box,
            email_box,
            workspace_box,
            requirements_box,
            status_box,
            history_box,
            session_state,
        ],
    )
    suggest_button.click(suggest_action, inputs=session_state, outputs=suggestion_box)
    demo.load(
        reset_task,
        inputs=task_picker,
        outputs=[
            objective_box,
            email_box,
            workspace_box,
            requirements_box,
            status_box,
            history_box,
            session_state,
        ],
    )


if __name__ == "__main__":
    demo.launch()
