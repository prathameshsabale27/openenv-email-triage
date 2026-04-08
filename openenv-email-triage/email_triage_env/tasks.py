"""Deterministic real-world email workflow tasks."""

from __future__ import annotations

from dataclasses import dataclass

from .models import InboxEmail


@dataclass(frozen=True)
class EmailOpsTask:
    """Single email triage task."""

    task_id: str
    difficulty: str
    objective: str
    email: InboxEmail
    expected_classification: str
    expected_queue: str
    expected_priority: str
    reply_required: bool
    expected_reply_keywords: tuple[str, ...]
    max_steps: int


TASKS: tuple[EmailOpsTask, ...] = (
    EmailOpsTask(
        task_id="easy-newsletter-cleanup",
        difficulty="easy",
        objective=(
            "Triage a promotional message. Review it, classify it correctly, route it to the "
            "right queue, set an appropriate priority, and submit the final triage decision."
        ),
        email=InboxEmail(
            sender="offers@deal-stream.example",
            subject="Last chance: claim your weekend shopping voucher",
            body=(
                "You have been selected for a weekend voucher worth $500. Click the secure link "
                "below before midnight to activate your reward."
            ),
        ),
        expected_classification="spam",
        expected_queue="junk",
        expected_priority="low",
        reply_required=False,
        expected_reply_keywords=(),
        max_steps=6,
    ),
    EmailOpsTask(
        task_id="medium-refund-followup",
        difficulty="medium",
        objective=(
            "Handle a customer support email about a duplicated invoice charge. Review the email, "
            "classify it, route it to the right queue, set urgency, draft a useful acknowledgment, "
            "and submit the case."
        ),
        email=InboxEmail(
            sender="april.nguyen@northstar-retail.example",
            subject="Charged twice for March invoice",
            body=(
                "Hi team, I was charged twice for invoice INV-2048 this morning. Please confirm "
                "whether a refund is being processed and let me know if you need anything from me."
            ),
        ),
        expected_classification="customer_support",
        expected_queue="billing",
        expected_priority="high",
        reply_required=True,
        expected_reply_keywords=("refund", "invoice", "investigate"),
        max_steps=8,
    ),
    EmailOpsTask(
        task_id="hard-executive-reschedule",
        difficulty="hard",
        objective=(
            "Handle an executive scheduling email. Review the request, classify it, route it, set "
            "the correct priority, draft a concrete reply, and submit a complete response."
        ),
        email=InboxEmail(
            sender="chief.of.staff@company.example",
            subject="Need to move tomorrow's APAC board prep",
            body=(
                "The CFO can no longer make tomorrow's APAC board prep. Please move the session to "
                "Thursday at 2 PM IST if that works for everyone, and reply to confirm once it is "
                "handled."
            ),
        ),
        expected_classification="scheduling",
        expected_queue="executive_ops",
        expected_priority="urgent",
        reply_required=True,
        expected_reply_keywords=("thursday", "2 pm", "confirm"),
        max_steps=8,
    ),
)

TASKS_BY_ID = {task.task_id: task for task in TASKS}
