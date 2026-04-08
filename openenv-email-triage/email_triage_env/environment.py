"""Local Gym-style environment with tuple step results."""

from __future__ import annotations

from itertools import cycle
from typing import Any
from uuid import uuid4

from .grader import TaskGrader
from .models import EnvironmentState, RewardModel, TriageAction, TriageObservation, WorkspaceState
from .tasks import TASKS, TASKS_BY_ID, EmailOpsTask


class EmailOpsEnvironment:
    """Multi-step email operations environment with incremental rewards."""

    benchmark_name = "email-triage-openenv"

    def __init__(self, tasks: tuple[EmailOpsTask, ...] = TASKS) -> None:
        self.tasks = tasks
        self._task_cycle = cycle(self.tasks)
        self.current_task: EmailOpsTask | None = None
        self._state = EnvironmentState()
        self._read_awarded = False
        self._submission_complete = False

    def reset(self, task_id: str | None = None) -> TriageObservation:
        if task_id is None:
            self.current_task = next(self._task_cycle)
        else:
            self.current_task = TASKS_BY_ID[task_id]

        self._state = EnvironmentState(
            episode_id=str(uuid4()),
            benchmark=self.benchmark_name,
            current_task_id=self.current_task.task_id,
            difficulty=self.current_task.difficulty,
            workspace=WorkspaceState(),
            reward_total=0.0,
            completed_requirements=[],
            failed_submissions=0,
            last_action_error=None,
            max_steps=self.current_task.max_steps,
        )
        self._read_awarded = False
        self._submission_complete = False
        return self._build_observation(reward_value=0.0, done=False)

    def state(self) -> EnvironmentState:
        return self._state.model_copy(deep=True)

    def close(self) -> None:
        return None

    def step(
        self, action: TriageAction | dict[str, Any]
    ) -> tuple[TriageObservation, RewardModel, bool, dict[str, Any]]:
        if self.current_task is None:
            raise RuntimeError("reset() must be called before step().")

        parsed_action = action if isinstance(action, TriageAction) else TriageAction(**action)
        self._state.step_count += 1
        self._state.last_action_error = None

        reward_delta = 0.0
        penalty = 0.0
        message = ""
        workspace = self._state.workspace

        if parsed_action.operation == "read_email":
            if not self._read_awarded:
                self._read_awarded = True
                reward_delta = 0.05
                message = "Email reviewed."
            else:
                reward_delta = -0.02
                penalty = 0.02
                self._state.last_action_error = "email_already_reviewed"
                message = "Repeated read_email action."

        elif parsed_action.operation == "classify":
            reward_delta, penalty, message = self._apply_choice(
                current_value=workspace.classification,
                new_value=parsed_action.value or "",
                expected_value=self.current_task.expected_classification,
                correct_reward=0.25,
                wrong_penalty=0.12,
                label="classification",
            )
            workspace.classification = parsed_action.value

        elif parsed_action.operation == "route":
            reward_delta, penalty, message = self._apply_choice(
                current_value=workspace.queue,
                new_value=parsed_action.value or "",
                expected_value=self.current_task.expected_queue,
                correct_reward=0.25,
                wrong_penalty=0.12,
                label="route",
            )
            workspace.queue = parsed_action.value

        elif parsed_action.operation == "set_priority":
            reward_delta, penalty, message = self._apply_choice(
                current_value=workspace.priority,
                new_value=parsed_action.value or "",
                expected_value=self.current_task.expected_priority,
                correct_reward=0.20,
                wrong_penalty=0.10,
                label="priority",
            )
            workspace.priority = parsed_action.value

        elif parsed_action.operation == "draft_reply":
            reward_delta, penalty, message = self._apply_reply(parsed_action.text or "")
            workspace.draft_reply = parsed_action.text or ""

        elif parsed_action.operation == "submit":
            progress_before = TaskGrader.score_fraction(self.current_task, workspace)
            if TaskGrader.is_complete(self.current_task, workspace):
                reward_delta = 0.25
                self._submission_complete = True
                message = "Task submitted successfully."
            else:
                reward_delta = -0.20
                penalty = 0.20
                self._state.failed_submissions += 1
                self._state.last_action_error = "submission_incomplete"
                message = (
                    f"Submission incomplete at progress {progress_before:.2f}; fill the remaining requirements first."
                )
        else:
            reward_delta = -0.25
            penalty = 0.25
            self._state.last_action_error = "unsupported_operation"
            message = f"Unsupported operation: {parsed_action.operation}"

        progress = TaskGrader.score_fraction(self.current_task, workspace)
        done = self._submission_complete

        if self._state.step_count >= self.current_task.max_steps and not done:
            done = True
            reward_delta -= 0.15
            penalty += 0.15
            message = (message + " " if message else "") + "Episode ended after hitting the step limit."
            if self._state.last_action_error is None:
                self._state.last_action_error = "max_steps_exceeded"

        self._state.reward_total = round(self._state.reward_total + reward_delta, 4)
        self._state.completed_requirements = TaskGrader.completed_requirements(
            self.current_task, workspace
        )

        reward = RewardModel(
            delta=round(reward_delta, 4),
            total=self._state.reward_total,
            progress=progress,
            penalty=round(penalty, 4),
            message=message or "Action processed.",
        )
        observation = self._build_observation(reward_value=reward.delta, done=done)
        info = {
            "task_id": self.current_task.task_id,
            "difficulty": self.current_task.difficulty,
            "score_fraction": progress,
            "last_action_error": self._state.last_action_error,
            "reward_model": reward.model_dump(),
        }
        return observation, reward, done, info

    def _apply_choice(
        self,
        *,
        current_value: str | None,
        new_value: str,
        expected_value: str,
        correct_reward: float,
        wrong_penalty: float,
        label: str,
    ) -> tuple[float, float, str]:
        if new_value == expected_value and current_value != expected_value:
            return correct_reward, 0.0, f"Correct {label} selected."
        if new_value == expected_value:
            self._state.last_action_error = f"{label}_already_correct"
            return -0.02, 0.02, f"{label.capitalize()} was already correct."

        self._state.last_action_error = f"incorrect_{label}"
        return -wrong_penalty, wrong_penalty, f"Incorrect {label}: '{new_value}'."

    def _apply_reply(self, draft_text: str) -> tuple[float, float, str]:
        assert self.current_task is not None
        workspace = self._state.workspace
        if not self.current_task.reply_required:
            self._state.last_action_error = "reply_not_needed"
            return -0.15, 0.15, "This task does not require a reply."

        previous = TaskGrader.reply_score(self.current_task, workspace.draft_reply)
        updated = TaskGrader.reply_score(self.current_task, draft_text)
        improvement = round(updated - previous, 4)

        if improvement > 0:
            reward = round(improvement * 0.35, 4)
            if updated == 1.0:
                return reward, 0.0, "Reply now covers all required details."
            return reward, 0.0, "Reply improved but is still incomplete."

        self._state.last_action_error = "reply_not_improved"
        return -0.05, 0.05, "Reply did not improve coverage of required details."

    def _build_observation(self, *, reward_value: float, done: bool) -> TriageObservation:
        assert self.current_task is not None
        pending = TaskGrader.pending_requirements(self.current_task, self._state.workspace)
        return TriageObservation(
            task_id=self.current_task.task_id,
            difficulty=self.current_task.difficulty,
            objective=self.current_task.objective,
            email=self.current_task.email,
            workspace=self._state.workspace.model_copy(deep=True),
            completed_requirements=list(self._state.completed_requirements),
            pending_requirements=pending,
            last_action_error=self._state.last_action_error,
            steps_remaining=max(0, self.current_task.max_steps - self._state.step_count),
            done=done,
            reward=reward_value,
            metadata={
                "benchmark": self.benchmark_name,
                "failed_submissions": self._state.failed_submissions,
            },
        )
