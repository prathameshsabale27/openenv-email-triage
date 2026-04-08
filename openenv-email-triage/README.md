---
title: Email Triage OpenEnv
tags:
  - openenv
  - reinforcement-learning
  - email-triage
sdk: docker
---

# Email Triage OpenEnv

Email Triage OpenEnv is a real-world workplace operations environment for the OpenEnv RL Challenge. The agent works through realistic inbox tasks that people actually do in support, operations, and scheduling workflows. Instead of a one-shot classifier, the environment requires a short trajectory of actions: inspect the email, populate a triage workspace, optionally draft a response, and submit a final decision.

## Motivation

Most production assistant workflows are not single-turn question answering. Human operators read a message, update multiple structured fields, draft a response, and only then finalize the task. This environment captures that pattern with deterministic tasks and reproducible graders.

## Repository Structure

```text
openenv-email-triage/
|-- app.py
|-- inference.py
|-- requirements.txt
|-- README.md
|-- openenv.yaml
|-- pyproject.toml
|-- server/
`-- openenv_email_triage/
    |-- env/
    `-- models/
```

## Environment Overview

Benchmark name: `email-triage-openenv`

Episodes use the following typed interfaces:

- Observation: `TriageObservation`
- Action: `TriageAction`
- Reward: `RewardModel`
- State: `EnvironmentState`

The local environment implements:

- `reset() -> observation`
- `step(action) -> (observation, reward, done, info)`
- `state() -> current_state`

The OpenEnv server adapter exposes the same task logic through the standard FastAPI runtime expected by `openenv validate`, while the root `app.py` provides a Gradio UI for Hugging Face Spaces.

## Observation Space

Each observation contains:

- `task_id`: deterministic task identifier
- `difficulty`: `easy`, `medium`, or `hard`
- `objective`: natural-language goal for the episode
- `email`: sender, subject, and body
- `workspace`: the agent's current classification, route, priority, and draft reply
- `completed_requirements`: requirements already satisfied
- `pending_requirements`: requirements still missing
- `last_action_error`: raw environment error string or `null`
- `steps_remaining`: remaining budget before forced termination
- `done` and `reward`: standard OpenEnv observation fields

## Action Space

`TriageAction` supports six operations:

- `read_email`
- `classify(value)` where `value` is one of `spam`, `customer_support`, `scheduling`
- `route(value)` where `value` is one of `junk`, `billing`, `executive_ops`
- `set_priority(value)` where `value` is one of `low`, `high`, `urgent`
- `draft_reply(text)`
- `submit()`

## Tasks

The benchmark ships with three deterministic tasks spanning increasing difficulty:

1. Easy: `easy-newsletter-cleanup`
   Promotional spam that should be classified, routed to junk, prioritized low, and submitted without a reply.
2. Medium: `medium-refund-followup`
   A customer billing request that must be routed to billing, prioritized high, and answered with a useful acknowledgment.
3. Hard: `hard-executive-reschedule`
   An executive scheduling request that requires correct routing, urgent priority, and a concrete confirmation reply.

## Graders

Each task includes a deterministic programmatic grader in [grader.py](/P:/openenv-email-triage/openenv_email_triage/env/grader.py). The grader returns a score in `[0.0, 1.0]` based on weighted completion of the required fields.

- Easy tasks score classification, route, and priority.
- Medium and hard tasks additionally score reply quality using keyword coverage.
- Reply scoring is deterministic and reproducible because it uses fixed keyword matching only.

## Reward Function

The reward function provides dense feedback through the trajectory instead of only at episode termination.

- First useful inspection via `read_email`: `+0.05`
- Correct first-time classification: `+0.25`
- Correct first-time routing: `+0.25`
- Correct first-time priority: `+0.20`
- Better reply coverage: proportional positive reward up to `+0.35`
- Successful final submission: `+0.25`
- Wrong choices, repeated loops, unnecessary replies, and incomplete submissions are penalized
- Exceeding the step budget adds an additional penalty

This makes the reward signal meaningful for RL because partial progress is visible immediately while repetitive or destructive behavior is discouraged.

## Setup

### Local Python

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

### Environment Variables

`inference.py` reads:

- `API_BASE_URL` with default `https://api.openai.com/v1`
- `MODEL_NAME` with default `gpt-4.1-mini`
- `HF_TOKEN` with no default and required at runtime

## Usage

### Run the Local Baseline Inference

```powershell
$env:HF_TOKEN = "your-token"
python inference.py
```

The script evaluates all three tasks in deterministic order and emits only the required challenge log lines:

- `[START]`
- `[STEP]`
- `[END]`

### Run the OpenEnv Server

```powershell
python -m server.app --host 0.0.0.0 --port 8000
```

### Run the Gradio UI

```powershell
python app.py
```

### Validate the Environment

```powershell
python -m openenv.cli validate .
python -m openenv.cli validate --url http://localhost:8000
```

## Docker

Build and run the environment container:

```powershell
docker build -t email-triage-openenv .
docker run --rm -p 8000:8000 email-triage-openenv
```

## Baseline Scores

Reference scores for the deterministic task logic:

- Oracle policy: easy `1.00`, medium `1.00`, hard `1.00`, average `1.00`
- `inference.py` uses an OpenAI model client and runs with `temperature=0`; run it with your `HF_TOKEN` to record a live baseline for the chosen model and endpoint.

## Hugging Face Spaces Notes

- The project is deployable as a Docker-based Hugging Face Space.
- Add the `openenv` tag when publishing the Space.
- Keep the primary submission Space running before submission so automated validation can reach the server.
