# Install + run documentation consolidation

**Type:** documentation refresh + consolidation (multi-host audit)
**Owner:** Coordinating agent (DGX) + per-host agents (DGX / Windows workstation / Jetson Orin Nano)
**Priority:** P2
**Estimate:** L (multi-host audit + 4 package READMEs + retire/merge of a handful of scattered docs; bounded by per-host availability)
**Branch:** `task/install-docs-consolidation`
**Blocked on:** [`windows-workstation-bringup`](windows-workstation-bringup.md) — that brief produces the authoritative Windows install path (PowerShell vs. WSL2) and the bridge runbook. We need its conclusions before we can author the Windows subsections of the package READMEs.

## Story

As a **new contributor (or future-me after six months away) standing in front of a fresh DGX / Windows workstation / Jetson** I want **one canonical place per package to learn how to install and run it on my host** so that **I don't have to grep across `DGX_SPARK_SETUP.md`, `HARNESS_DATA_CAPTURE.md`, `example_commands_cheatsheet.md`, `INTEGRATION_SIM_IN_THE_LOOP.md`, and four package READMEs (each with its own partial Install section) to figure out what actually works today.**

## Motivation

Install + run knowledge is currently scattered:

| Doc | What it covers | Freshness |
|---|---|---|
| [`Readme.md`](../../../../Readme.md) | Landing page + role-of-the-system table | OK as a landing page |
| [`docs/DGX_SPARK_SETUP.md`](../../../DGX_SPARK_SETUP.md) | DGX-side Isaac Lab bringup | **Stale** — operator reports it does not match what works today |
| [`docs/HARNESS_DATA_CAPTURE.md`](../../../HARNESS_DATA_CAPTURE.md) | Harness env setup + capture commands | Fresh (just landed in PR #63); duplicates some setup steps that should live in the package README |
| [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../../INTEGRATION_SIM_IN_THE_LOOP.md) | Cross-host sim-in-the-loop runbook | Unknown freshness; some env setup overlaps with DGX_SPARK_SETUP |
| [`docs/D555_IMU_KERNEL_FIX.md`](../../../D555_IMU_KERNEL_FIX.md) | Jetson camera kernel quirk | Standalone procedure; correct home — link from strafer_ros README |
| [`docs/example_commands_cheatsheet.md`](../../../example_commands_cheatsheet.md) | One-liners across the whole stack | Mixed; each section ages independently |
| [`source/strafer_lab/README.md`](../../../../source/strafer_lab/README.md) | strafer_lab features + contracts; has an `## Install` section already | Install section may be stale; no per-platform subsections |
| [`source/strafer_ros/README.md`](../../../../source/strafer_ros/README.md) | strafer_ros bringup; has an `## Install` section | Same |
| [`source/strafer_vlm/README.md`](../../../../source/strafer_vlm/README.md) | strafer_vlm service + fine-tuning; has an `## Install` section | Same |
| [`source/strafer_autonomy/README.md`](../../../../source/strafer_autonomy/README.md) | Planner + executor; has an `## Install` section | Same |

The "package README is the source of truth" rule is already stated in
`Readme.md` ("Each package README is the source of truth for that
package's features, contracts, setup, commands, design, and testing")
— but the actual install + run content is fragmented across the
right-hand column of that table. This brief realigns reality to that
rule.

**Environments are fragmented too** — though one fork just closed. The live
set is now **three**: `env_isaaclab3` (3.12; Isaac Sim/Lab + `pxr` + CUDA
torch 2.10 + lerobot + warp/onnx — training, the sim bridge, **and all
strafer_lab tests**), `env_infinigen` (3.11; scene-gen), and `.venv_vlm`
(3.12; CUDA torch 2.11 + transformers 5.x — the VLM/planner service + its
tests). Three others are gone or going: `env_isaaclab` (3.11, superseded by
`env_isaaclab3`) and `blender-build-py311` (a one-time Blender-build env whose
output binary is standalone) are dead; **`.venv_harness` is retired** — the
test-tree-unification PR confirmed lerobot 0.5.1 coexists with
`env_isaaclab3`'s CUDA torch (its old **CPU**-torch split was the only reason
it existed), so the harness suite folded into `env_isaaclab3` and gained
`pxr`. The two remaining splits are both **forced, not incidental**:
Infinigen's 3.11 pin, and — the deeper one — `.venv_vlm` is kept **by design**
because Isaac Sim's *compiled* torch is a hard floor (`env_isaaclab3` can't
move off torch 2.10 without risking the sim) while the VLM/LLM stack wants the
fast-moving ceiling (newer transformers/torch per newer models). A canonical
install story must name which env is for what, why it's separate, and how to
recreate it — and prune the dead ones.

## Acceptance

Ship a documentation pass that meets all of:

- [ ] Each package README's `## Install` section is **refreshed against
  a live install on each supported host** and now has explicit
  `### Linux (DGX Spark / x86_64 workstation)` and `### Windows
  (workstation)` subsections where the package runs on both. Each
  subsection includes:
  - Prereqs (OS version verified, GPU + driver, Python version, conda or
    venv name conventionally used)
  - Step-by-step install commands actually run during the audit
  - Smoke-test command + expected output
  - Known platform limitations (e.g. "DGX Spark: no SkillGen / OpenXR /
    JAX-GPU / Livestream"; "Windows: Isaac Lab 3 is experimental")
  - **Freshness footer**: `_Last verified <YYYY-MM-DD> on <hostname or
    spec>._`
- [ ] Each package README has (or gains) a `## Run` section right after
  `## Install` listing the package's main entry points + canonical
  invocations. The cheatsheet stops being the source of truth for
  these; it keeps only cross-package "what to type during a live demo"
  one-liners and points at the package READMEs for everything else.
- [ ] `Readme.md` (top-level landing) is updated so the per-package
  links in the role-of-the-system table point at the new Install +
  Run anchors specifically (e.g. `[Install](source/strafer_lab/README.md#install)`).
  Cross-host architecture and the call-path narrative stay on the
  landing page; install detail stays in the READMEs.
- [ ] `docs/DGX_SPARK_SETUP.md` is **retired** — the content moves into
  `strafer_lab/README.md`'s Linux subsection (the DGX-specific knobs
  + caveats live there, alongside any other Linux info). Leave a
  redirect note in the old file for one PR cycle, then delete in a
  follow-up cleanup.
- [ ] `docs/INTEGRATION_SIM_IN_THE_LOOP.md` is reviewed and either
  refreshed in-place (if it stays a runbook) or merged into the
  strafer_lab + strafer_ros READMEs' Run sections. Same for
  `docs/HARNESS_DATA_CAPTURE.md` — keep as a deep-dive harness guide
  linked from `strafer_lab/README.md#run`, but verify nothing in it
  duplicates package-README install content.
- [ ] `docs/D555_IMU_KERNEL_FIX.md` stays as its own file and is linked
  from `strafer_ros/README.md`'s Linux subsection.
- [ ] `docs/example_commands_cheatsheet.md` is pruned to only
  cross-package live-demo one-liners; each subsection that exists
  today is either kept (as a live-demo crib sheet) or replaced with a
  one-line pointer at the responsible package README's Run section.
- [ ] No installation or run command is documented in more than one
  place after the pass. Where two sources would otherwise agree, one
  becomes the source and the other becomes a link.
- [ ] **Environment topology documented + rationalized.** `repo-topology.md`
  (and the relevant package READMEs) name the live conda/venv set, what each
  is for, the hard constraint that keeps it separate, and a copy-paste
  recreate command. **Prune:** delete the two dead envs (`env_isaaclab`,
  `blender-build-py311`) **and** the now-retired `.venv_harness` dir (after
  confirming nothing live references them — the Makefile/READMEs already
  don't). The **`.venv_harness` consolidation probe is resolved + executed**
  (test-tree-unification PR): lerobot 0.5.1 coexists with `env_isaaclab3`'s
  CUDA torch 2.10, so the harness suite runs there (211 tests incl. the 27
  `pxr` postprocess tests that previously skipped), the entry points are
  `make test-lab` / `make test-lab-pure`, and the env count dropped to three.
  **Record the `.venv_vlm` decision: kept by design** — cadence isolation
  (Isaac Sim's compiled-torch floor vs the VLM/LLM stack's fast-moving
  transformers/torch ceiling), so no further test-env consolidation is
  pursued and the standalone `python-env-topology` fallback brief is **not**
  needed.
- [ ] If your work invalidates a fact in any referenced context module
  or guide, update those in the same commit (per
  [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)).

## Approach — multi-agent audit

The accuracy bar requires audit-on-host-from-fresh-shell rather than
desk research. A coordinating agent on the DGX runs the brief; per-host
agents do the audits. The user is the transport layer between sessions
(no Claude-to-Claude direct comms; the user copies prompts + reports).

### Topology

```
                       ┌─────────────────────────────────┐
                       │     Coordinating agent (DGX)    │
                       │                                 │
                       │  - Drafts per-host prompts      │
                       │  - Integrates returned reports  │
                       │  - Flags inconsistencies        │
                       │  - Writes the canonical README  │
                       │    updates + commits the PR     │
                       └────────┬───────────┬────────────┘
                                │           │
              (user pastes      │           │      (user pastes
               prompt; user     │           │       report back
               relays report)   │           │       into coord)
                                ▼           ▼
              ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
              │ DGX Spark Linux     │ │ Windows workstation │ │ Jetson Orin Nano    │
              │ Claude Code session │ │ Claude Code session │ │ Claude Code session │
              │                     │ │                     │ │                     │
              │ Audits:             │ │ Audits:             │ │ Audits:             │
              │ - strafer_lab       │ │ - strafer_lab       │ │ - strafer_ros       │
              │ - strafer_vlm       │ │ - strafer_vlm       │ │ - strafer_autonomy  │
              │ - strafer_autonomy  │ │ - strafer_autonomy  │ │   (executor side)   │
              │   (planner side)    │ │   (planner side)    │ │ - strafer_vlm grpc  │
              │                     │ │ Depends on:         │ │   client (no svc)   │
              │                     │ │ windows-workstation-│ │                     │
              │                     │ │ bringup conclusions │ │                     │
              └─────────────────────┘ └─────────────────────┘ └─────────────────────┘
```

Strafer_shared is library-only (no install of its own); it gets covered
implicitly by the packages that depend on it.

### Per-host agent return format (contract)

Each per-host agent returns a structured Markdown report so the
coordinator can integrate without ambiguity. The coordinator's prompt
spells this out explicitly. Required sections per package audited:

```markdown
## <package_name> on <host_name>

### Environment snapshot
- OS: <`lsb_release -a` or `winver`>
- Kernel / build: <`uname -a` or PowerShell equivalent>
- GPU + driver: <`nvidia-smi --query-gpu=name,driver_version --format=csv`>
- Python: <`python --version`> in env `<env_name>`
- Other relevant tools: <ROS distro, conda version, etc.>

### Prereqs verified
- [ ] <prereq 1> — passed via <command>
- [ ] <prereq 2> — passed via <command>

### Install steps that worked
```bash
# Exact commands run, in order. Include `conda activate <env>` lines.
```

### Smoke test
```bash
# The single command that proves the install works
```

Expected output:
```
<paste actual output verbatim>
```

### Known limitations / quirks on this host
- <quirk 1, with workaround if known>
- <quirk 2>

### Discrepancies vs. current docs
- <file path>:<line range> — current text says X, reality is Y

### Verified on
<YYYY-MM-DD> on `<hostname>` (`<short hardware spec>`)
```

### Coordinator responsibilities (beyond stitching)

When the coordinator integrates reports it MUST:

1. **Flag inconsistencies between hosts** — e.g. Windows agent says
   `python 3.11`, Linux agent says `python 3.12` in the same env name.
   Surface to user; don't silently pick one.
2. **Flag silent assumptions** — if a per-host report says "I assumed
   git clone is already done", document that assumption in the README.
3. **Refuse to author** Windows subsections until
   [`windows-workstation-bringup`](windows-workstation-bringup.md) is
   shipped. Author the Linux + Jetson parts first; land that as a PR;
   pick up the Windows pass when its dependency clears.
4. **Drop the freshness footer at the bottom of each refreshed section**
   with the date and host hardware string copied verbatim from the
   per-host report's "Verified on" line. Future readers should be able
   to grep `Last verified` and date-sort.

### Per-host agent prompt sketch

The coordinator drafts the actual prompts at execution time, but they
should follow this skeleton so prompts stay consistent across hosts:

> You are auditing the install + run instructions for `<package_name>`
> on `<host_name>` for the Sim2RealLab repo. The current README is at
> `<path>`. Your job is to:
> 1. Read the current README's Install + Run sections.
> 2. Wipe your conda env (or document why you can't) and rebuild from
>    scratch following ONLY the README's instructions verbatim.
> 3. Note every command that didn't work, every prereq that was
>    silently assumed, and every version that differs from what the
>    README claims.
> 4. Run the smoke test commands the README documents.
> 5. Return a report in the format below. Be terse but
>    copy-paste-accurate.
>
> <paste the return-format contract from this brief>
>
> The user will copy your report back into a coordinating agent on the
> DGX which integrates it into the canonical README pass.

## Approach — content shape per package

The coordinator should hold to this shape so the four READMEs feel
consistent after the pass:

```markdown
## Install

### Linux (DGX Spark / x86_64 workstation)
<prereqs, install commands, smoke test, limitations>
_Last verified 2026-MM-DD on `<hostname>` (`<spec>`)._

### Windows (workstation)
<same shape>
_Last verified 2026-MM-DD on `<hostname>` (`<spec>`)._

### Jetson Orin Nano   *(strafer_ros only)*
<same shape; link to D555_IMU_KERNEL_FIX.md>
_Last verified 2026-MM-DD on `<hostname>` (`<spec>`)._

## Run

### Canonical entry points
- `<command>` — <what it does>
- `<command>` — <what it does>

### Deep-dive runbooks
- [`HARNESS_DATA_CAPTURE.md`](../../../docs/HARNESS_DATA_CAPTURE.md) — full teleop capture workflow
- ...
```

## Out of scope

- Rewriting the package READMEs' Features / Contracts / Design / Testing
  sections. This pass only touches Install + Run.
- Authoring the Windows install procedure itself — that's
  [`windows-workstation-bringup`](windows-workstation-bringup.md)'s
  job. This brief consumes the output.
- Replacing `D555_IMU_KERNEL_FIX.md` — that's a specific procedure that
  earns its own file. We just link to it from `strafer_ros/README.md`.
- A CI check that re-verifies docs against a live install. That's a
  potential follow-up (`docs-freshness-ci` brief if a drift incident
  surfaces) but not in scope here.

## Triggered by

User observation during PR #63 review: "I'm noticing the
`docs/DGX_SPARK_SETUP.md` is out of date, and should be replaced. I'm
also noticing installation and run guide instructions in various places
in the project — each of which I'm unsure how fresh are. I think there
should be one place for installation and runnable instructions for each
platform and purpose."
