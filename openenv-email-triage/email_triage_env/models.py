"""Typed models for the email triage environment."""

from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, model_validator


VALID_CLASSIFICATIONS = ("spam", "customer_support", "scheduling")
VALID_QUEUES = ("junk", "billing", "executive_ops")
VALID_PRIORITIES = ("low", "high", "urgent")
VALID_OPERATIONS = (
    "read_email",
    "classify",
    "route",
    "set_priority",
    "draft_reply",
    "submit",
)


class InboxEmail(BaseModel):
    """The email currently being processed."""

    sender: str = Field(..., description="Email sender address or display name.")
    subject: str = Field(..., description="Email subject line.")
    body: str = Field(..., description="Full email body text.")


class WorkspaceState(BaseModel):
    """Editable workspace the agent manipulates over the episode."""

    classification: str | None = Field(default=None)
    queue: str | None = Field(default=None)
    priority: str | None = Field(default=None)
    draft_reply: str = Field(default="")


class RewardModel(BaseModel):
    """Structured reward details returned by the local environment."""

    delta: float = Field(..., ge=-1.0, le=1.0)
    total: float = Field(...)
    progress: float = Field(..., ge=0.0, le=1.0)
    penalty: float = Field(default=0.0, ge=0.0)
    message: str = Field(...)


class TriageAction(Action):
    """Single agent action taken inside the episode."""

    operation: Literal[
        "read_email",
        "classify",
        "route",
        "set_priority",
        "draft_reply",
        "submit",
    ] = Field(..., description="The operation to perform.")
    value: str | None = Field(
        default=None,
        description="Value for classify, route, or set_priority operations.",
    )
    text: str | None = Field(
        default=None,
        description="Reply content used by the draft_reply operation.",
    )

    @model_validator(mode="after")
    def validate_payload(self) -> "TriageAction":
        if self.operation in {"classify", "route", "set_priority"} and not self.value:
            raise ValueError(f"operation '{self.operation}' requires a value")
        if self.operation == "draft_reply" and not self.text:
            raise ValueError("operation 'draft_reply' requires text")
        return self


class TriageObservation(Observation):
    """Observation returned after reset and each step."""

    task_id: str = Field(...)
    difficulty: Literal["easy", "medium", "hard"] = Field(...)
    objective: str = Field(..., description="Task objective shown to the agent.")
    email: InboxEmail = Field(...)
    workspace: WorkspaceState = Field(...)
    completed_requirements: list[str] = Field(default_factory=list)
    pending_requirements: list[str] = Field(default_factory=list)
    last_action_error: str | None = Field(default=None)
    steps_remaining: int = Field(..., ge=0)


class EnvironmentState(State):
    """Full serializable environment state."""

    benchmark: str = Field(default="email-triage-openenv")
    current_task_id: str | None = Field(default=None)
    difficulty: str | None = Field(default=None)
    workspace: WorkspaceState = Field(default_factory=WorkspaceState)
    reward_total: float = Field(default=0.0)
    completed_requirements: list[str] = Field(default_factory=list)
    failed_submissions: int = Field(default=0, ge=0)
    last_action_error: str | None = Field(default=None)
    max_steps: int = Field(default=0, ge=0)


def compact_action_string(action: TriageAction) -> str:
    """Render a stable single-line action string for logs."""

    if action.operation == "draft_reply":
        text = (action.text or "").replace("\n", " ").replace("'", "\\'")
        return f"draft_reply('{text}')"
    if action.value is not None:
        value = action.value.replace("'", "\\'")
        return f"{action.operation}('{value}')"
    return f"{action.operation}()"


def reward_to_info(reward: RewardModel) -> dict[str, Any]:
    """Serialize reward details into an info payload."""

    return reward.model_dump()
