# Ownership boundaries

The repo is worked on by two parallel agents (one per host). Each
agent owns a subset of the source tree. **You do not edit files
outside your lane**, and **you do not talk to the other agent
directly** — the operator (the human user) relays observations
between you.

If a task brief is addressed to "Either" (`Owner: either`), pick the
host whose lane the change lives in.

## DGX agent's lane

You may read and modify:

| Path | What it is |
|------|------------|
| `source/strafer_lab/` | Isaac Sim envs, RL policies, ROS 2 sim bridge, sim-in-the-loop harness |
| `source/strafer_vlm/` | VLM service (HTTP, DGX-side) |
| `source/strafer_autonomy/strafer_autonomy/planner/` | LLM planner |
| `source/strafer_autonomy/strafer_autonomy/clients/planner_client.py` | Executor's HTTP client → planner |
| `source/strafer_autonomy/strafer_autonomy/clients/vlm_client.py` | Executor's HTTP client → VLM |
| `source/strafer_autonomy/strafer_autonomy/semantic_map/` | Semantic-map data structures |
| `source/strafer_autonomy/strafer_autonomy/cli.py` | `strafer-autonomy-cli` entry point |
| `source/strafer_autonomy/strafer_autonomy/schemas/` | Schema definitions |
| `Scripts/` | Operator-facing DGX entry points |
| `env_setup.sh`, `.env`, `.env.example` | Operator-tuned host paths |
| Repo-root `Makefile` | DGX targets (test-dgx, sim-bridge, sim-bridge-gui, serve-vlm, serve-planner) |
| `docs/` | Documentation (excluding the Jetson-specific runbooks below if any are added later) |

## Jetson agent's lane

You may read and modify:

| Path | What it is |
|------|------------|
| `source/strafer_ros/` | All ROS 2 packages: bringup, slam, nav, perception, driver, msgs, description |
| `source/strafer_autonomy/strafer_autonomy/executor/` | Mission executor (`strafer-executor`) |
| `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py` | Executor's ROS client (Nav2, rotate, scan, etc.) |
| `source/strafer_shared/` | Cross-host shared constants — see [Shared boundary](#shared-boundary) below |
| Jetson-side test directories under each `strafer_ros/*` package |

## Shared boundary

`source/strafer_shared/strafer_shared/constants.py` is **append-only
across the boundary**: either lane may add new constants, but neither
should remove or rename existing ones without coordinating through
the operator. The whole point of `strafer_shared` is one source of
truth for chassis specs, sensor specs, and topic conventions —
ad-hoc edits create sim-to-real drift.

## Off-limits, regardless of lane

- `docs/archive/` — does not exist anymore (deleted in `ded56ea`).
  Don't recreate it. Closed task briefs are deleted, not archived.
- Raw Infinigen-generated USDC files under `Assets/generated/scenes/`
  — outputs of `prep_room_usds.py`, not source. Treat as build
  artifacts.
- Anything under `logs/` — RL run output, gitignored.

## The peer-agent rule

You do not ping the other agent directly. The operator may paste
observations from one agent's session into the other's, or write a
brief in `scratch_pad.md` / `dgx_scratch_pad.md` summarizing
cross-lane context for you. Operate as if your peer is a black box
whose lane you must not touch.

When you write a task brief whose work crosses the boundary,
either:

1. Pick a primary owner and note in `## Out of scope` what the
   peer-side work is, OR
2. Mark it `Owner: either` and let whichever lane the heart of the
   change lives in pick it up.

## Mode of work

Each agent's session is bootstrapped by a one-shot brief that names
its host, its lane, and the specific task at hand. After the brief is
read, the agent is expected to operate independently within the lane
until the task is shipped. **Don't broaden scope mid-session** — if
new work is discovered that needs another lane, write a follow-up
task brief in `docs/tasks/` for the appropriate owner and call it
out in your end-of-session summary.

## When the boundary itself needs to move

The lane lists above are not eternal. If a refactor reasonably
reorganizes responsibilities (e.g., a new package emerges that
straddles current owners), update this module in the same commit
that lands the refactor. That's the maintenance contract for context
modules described in
[`README.md`](README.md#maintenance-contract).
