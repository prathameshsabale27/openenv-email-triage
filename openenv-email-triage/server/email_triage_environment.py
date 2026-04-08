"""OpenEnv server adapter for the local email triage environment."""

from __future__ import annotations

from pathlib import Path

from openenv.core.env_server.interfaces import Environment, EnvironmentMetadata

from email_triage_env.environment import EmailOpsEnvironment
from email_triage_env.models import EnvironmentState, TriageAction, TriageObservation


class OpenEnvEmailTriageEnvironment(Environment[TriageAction, TriageObservation, EnvironmentState]):
    """OpenEnv-compatible wrapper around the local tuple-based environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._env = EmailOpsEnvironment()

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **kwargs,
    ) -> TriageObservation:
        observation = self._env.reset(task_id=task_id)
        if episode_id:
            self._env._state.episode_id = episode_id
        return observation

    def step(
        self,
        action: TriageAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> TriageObservation:
        observation, reward, done, info = self._env.step(action)
        observation.done = done
        observation.reward = reward.delta
        observation.metadata.update(
            {
                "score_fraction": info["score_fraction"],
                "reward_model": reward.model_dump(),
            }
        )
        return observation

    @property
    def state(self) -> EnvironmentState:
        return self._env.state()

    def get_metadata(self) -> EnvironmentMetadata:
        readme_path = Path(__file__).resolve().parent.parent / "README.md"
        readme_content = readme_path.read_text(encoding="utf-8") if readme_path.exists() else None
        return EnvironmentMetadata(
            name="email-triage-openenv",
            description="A multi-step workplace email triage environment with incremental rewards.",
            version="1.0.0",
            readme_content=readme_content,
        )

    def close(self) -> None:
        self._env.close()
