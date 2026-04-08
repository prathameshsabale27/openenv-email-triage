"""Typed client for remote OpenEnv deployment."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from openenv_email_triage.models.schemas import EnvironmentState, TriageAction, TriageObservation


class EmailTriageEnvClient(EnvClient[TriageAction, TriageObservation, EnvironmentState]):
    """Client for a running email triage OpenEnv server."""

    def _step_payload(self, action: TriageAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[TriageObservation]:
        observation = TriageObservation(**payload["observation"])
        reward = payload.get("reward")
        done = payload.get("done", False)
        return StepResult(observation=observation, reward=reward, done=done)

    def _parse_state(self, payload: dict[str, Any]) -> EnvironmentState:
        return EnvironmentState(**payload)
