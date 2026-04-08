"""Deterministic graders for each email triage task."""

from __future__ import annotations

from .models import WorkspaceState
from .tasks import EmailOpsTask


class TaskGrader:
    """Programmatic, deterministic grading helpers."""

    @staticmethod
    def reply_score(task: EmailOpsTask, draft_reply: str) -> float:
        if not task.reply_required:
            return 1.0 if not draft_reply.strip() else 0.0

        reply = draft_reply.lower()
        if not reply:
            return 0.0

        matches = sum(1 for keyword in task.expected_reply_keywords if keyword in reply)
        return round(matches / max(1, len(task.expected_reply_keywords)), 4)

    @classmethod
    def requirement_status(
        cls, task: EmailOpsTask, workspace: WorkspaceState
    ) -> dict[str, float]:
        status = {
            "classification": 1.0
            if workspace.classification == task.expected_classification
            else 0.0,
            "route": 1.0 if workspace.queue == task.expected_queue else 0.0,
            "priority": 1.0 if workspace.priority == task.expected_priority else 0.0,
        }
        if task.reply_required:
            status["reply"] = cls.reply_score(task, workspace.draft_reply)
        return status

    @classmethod
    def score_fraction(cls, task: EmailOpsTask, workspace: WorkspaceState) -> float:
        status = cls.requirement_status(task, workspace)
        if task.reply_required:
            score = (
                status["classification"] * 0.25
                + status["route"] * 0.20
                + status["priority"] * 0.20
                + status["reply"] * 0.35
            )
        else:
            score = (
                status["classification"] * 0.4
                + status["route"] * 0.3
                + status["priority"] * 0.3
            )
        return round(min(1.0, max(0.0, score)), 4)

    @classmethod
    def completed_requirements(
        cls, task: EmailOpsTask, workspace: WorkspaceState
    ) -> list[str]:
        status = cls.requirement_status(task, workspace)
        completed: list[str] = []
        if status["classification"] == 1.0:
            completed.append("classification")
        if status["route"] == 1.0:
            completed.append("route")
        if status["priority"] == 1.0:
            completed.append("priority")
        if task.reply_required and status.get("reply", 0.0) == 1.0:
            completed.append("reply")
        return completed

    @classmethod
    def pending_requirements(
        cls, task: EmailOpsTask, workspace: WorkspaceState
    ) -> list[str]:
        required = ["classification", "route", "priority"]
        if task.reply_required:
            required.append("reply")
        completed = set(cls.completed_requirements(task, workspace))
        return [item for item in required if item not in completed]

    @classmethod
    def is_complete(cls, task: EmailOpsTask, workspace: WorkspaceState) -> bool:
        return cls.score_fraction(task, workspace) == 1.0
