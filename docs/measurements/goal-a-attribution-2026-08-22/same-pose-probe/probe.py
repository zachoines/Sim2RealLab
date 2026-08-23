#!/usr/bin/env python3
"""Same-pose probe (SCRATCH script — not a repo file).

Reproduces the arm3 robot-side capture pose in pure sim on
Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0 @ seed 42, dumps
gym-assembled observations at that pose, and runs the v2 (and v1) exported
policies closed-loop from it.

App-launch boilerplate copied from scripts/play_strafer_navigation.py.
Teleport / injection / dump patterns per verified repo anchors:
  - yaw-pin + terminations-disable mirror: run_sim_in_the_loop.py:717-742
  - teleport: strafer_lab/tools/grounding_frame_provider.py:101 (_teleport_robot)
  - referent injection: mdp/commands.py _goal + events.py:400-404 precedent
  - obs dump: strafer_lab/bridge/obs_dump_terms.py:116 make_bridge_obs_dumper
  - policy load: play_strafer_navigation.py:205-215 (_load_exported_policy pattern)
"""

import argparse
import json
import math
import os
import sys
import traceback

OUT_DIR = "docs/measurements/goal-a-attribution-2026-08-22/same-pose-probe"

TASK = "Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0"
SEED = 42
PIN_YAW = 0.0  # mirrors run_sim_in_the_loop --pin-yaw default (RNG-relevant)
START_MAP = (-0.499, -0.451)   # robot mission-start pose, MAP frame
START_YAW = 2.22               # rad (map yaw == world yaw; spawn yaw pinned 0)
REF_MAP = (-0.12990365596194792, 0.3428494265663703)  # subgoal referent, MAP
GOAL_MAP = (-2.0, 2.25)        # mission goal, MAP
SPAWN_Z = 0.1                  # matches reset_robot_state fixed z (events.py:62)
SETTLE_STEPS = 2               # grounding_frame_provider DEFAULT_GROUNDING_WARMUP_STEPS
DUMP_STEPS = 30
CLOSED_LOOP_S = 15.0
V2_PT = os.path.expanduser("~/Workspace/Sim2RealLab/models/strafer_depth_subgoal_v2_998.pt")
V1_PT = os.path.expanduser("~/Workspace/Sim2RealLab/models/strafer_depth_subgoal_v1.pt")

parser = argparse.ArgumentParser(description="Same-pose sim probe (scratch)")
from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

EXIT_CODE = 1
env = None

def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))

try:
    import gymnasium as gym
    import torch
    import warp as wp

    from isaaclab_tasks.utils import parse_env_cfg

    import strafer_lab  # noqa: F401  (registers envs)
    from strafer_lab.tools.grounding_frame_provider import _teleport_robot
    from strafer_lab.bridge.obs_dump_terms import make_bridge_obs_dumper
    from strafer_lab.tasks.navigation.mdp import (
        goal_position_relative,
        goal_distance,
        goal_heading_to_goal,
        goal_heading_relative,
    )

    def TT(x):
        return x if isinstance(x, torch.Tensor) else wp.to_torch(x)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- cfg (mirror RNG-relevant sim-in-the-loop overrides) ----
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    print(f"[probe] cfg default seed = {env_cfg.seed}")
    env_cfg.seed = SEED

    reset_term = getattr(env_cfg.events, "reset_robot", None)
    assert reset_term is not None and getattr(reset_term, "params", None)
    params = reset_term.params
    if "yaw_range" in params:
        params["yaw_range"] = (PIN_YAW, PIN_YAW)
        print(f"[probe] reset yaw pinned to {PIN_YAW:.3f} rad (yaw_range)")
    elif "pose_range" in params and isinstance(params["pose_range"], dict):
        params["pose_range"]["yaw"] = (PIN_YAW, PIN_YAW)
        print(f"[probe] reset yaw pinned to {PIN_YAW:.3f} rad (pose_range)")
    else:
        raise RuntimeError("no yaw knob on reset_robot params")

    disabled = []
    for name, term in list(vars(env_cfg.terminations).items()):
        if name.startswith("_") or term is None:
            continue
        setattr(env_cfg.terminations, name, None)
        disabled.append(name)
    print(f"[probe] terminations disabled: {disabled}")
    print(f"[probe] decimation={env_cfg.decimation} sim.dt={env_cfg.sim.dt} "
          f"render_interval={env_cfg.sim.render_interval}")

    # ---- env + ONE reset ----
    env = gym.make(TASK, cfg=env_cfg)
    uw = env.unwrapped
    obs, extras = env.reset()
    print("[probe] reset done")

    robot = uw.scene["robot"]
    data = robot.data
    pos_attr = "root_link_pos_w" if hasattr(data, "root_link_pos_w") else "root_pos_w"
    quat_attr = "root_link_quat_w" if hasattr(data, "root_link_quat_w") else "root_quat_w"

    def read_pose():
        p = TT(getattr(data, pos_attr))[0].detach().cpu().double()
        q = TT(getattr(data, quat_attr))[0].detach().cpu().double()  # XYZW
        qx, qy, qz, qw = q[0].item(), q[1].item(), q[2].item(), q[3].item()
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return [p[0].item(), p[1].item(), p[2].item()], yaw

    anchor_pos, anchor_yaw = read_pose()
    env_origin = uw.scene.env_origins[0].detach().cpu().double().tolist()
    print(f"[probe] ANCHOR pos_w = ({anchor_pos[0]!r}, {anchor_pos[1]!r}, {anchor_pos[2]!r})")
    print(f"[probe] ANCHOR yaw_w = {anchor_yaw!r}  (pin target {PIN_YAW})")
    print(f"[probe] env_origin = {env_origin!r}  pos_attr={pos_attr}")

    act_shape = tuple(int(d) for d in uw.action_manager.action.shape)
    zero_action = torch.zeros(act_shape, device=uw.device)
    step_dt = float(uw.step_dt)
    print(f"[probe] action shape={act_shape} step_dt={step_dt}")

    def obs_policy(o):
        if hasattr(o, "keys") and "policy" in o.keys():
            return o["policy"]
        return o

    print(f"[probe] policy obs dim = {tuple(obs_policy(obs).shape)}")
    try:
        terms = uw.observation_manager.active_terms["policy"]
        dims = uw.observation_manager.group_obs_term_dim["policy"]
        term_layout = [[t, [int(x) for x in (d if hasattr(d, "__len__") else [d])]]
                       for t, d in zip(terms, dims)]
    except Exception as e:  # noqa: BLE001
        term_layout = f"unavailable: {e}"
    print(f"[probe] policy term layout: {term_layout}")

    target_world = (anchor_pos[0] + START_MAP[0], anchor_pos[1] + START_MAP[1])
    ref_world = (anchor_pos[0] + REF_MAP[0], anchor_pos[1] + REF_MAP[1])
    goal_world = (anchor_pos[0] + GOAL_MAP[0], anchor_pos[1] + GOAL_MAP[1])
    env_ids = torch.tensor([0], device=uw.device, dtype=torch.long)

    def t_sim_now():
        t = getattr(uw.sim, "current_time", None)
        if t is None:
            t = uw.common_step_counter * step_dt
        return float(t)

    def place_and_settle():
        """Teleport env 0 to the capture pose, inject referent, settle."""
        _teleport_robot(
            robot=robot, scene=uw.scene, env_ids=env_ids, device=uw.device,
            x=target_world[0] - env_origin[0],
            y=target_world[1] - env_origin[1],
            yaw=START_YAW, spawn_z=SPAWN_Z, torch=torch,
        )
        term = uw.command_manager.get_term("goal_command")
        term._goal[0, 0] = ref_world[0]
        term._goal[0, 1] = ref_world[1]
        term._goal[0, 2] = 0.0
        o = None
        for _ in range(SETTLE_STEPS):
            o, _r, _te, _tr, _in = env.step(zero_action.clone())
        # re-assert referent in case any manager touched it during settle
        term._goal[0, 0] = ref_world[0]
        term._goal[0, 1] = ref_world[1]
        term._goal[0, 2] = 0.0
        return o

    # ---- placement + readback ----
    obs = place_and_settle()
    achieved_pos, achieved_yaw = read_pose()
    quartet = {
        "goal_position_relative_raw_m": goal_position_relative(uw, "goal_command")[0].detach().cpu().tolist(),
        "goal_distance_raw_m": goal_distance(uw, "goal_command")[0].detach().cpu().tolist(),
        "goal_heading_to_goal_raw_rad": goal_heading_to_goal(uw, "goal_command")[0].detach().cpu().tolist(),
        "goal_heading_relative_raw_rad": goal_heading_relative(uw, "goal_command")[0].detach().cpu().tolist(),
    }
    print(f"[probe] PLACED pos_w={achieved_pos!r} yaw_w={achieved_yaw!r}")
    print(f"[probe] target_world={target_world!r} target_yaw={START_YAW}")
    print(f"[probe] quartet(clean terms): {json.dumps(quartet)}")

    goal_bearing_body = _wrap(math.atan2(goal_world[1] - achieved_pos[1],
                                         goal_world[0] - achieved_pos[0]) - achieved_yaw)
    print(f"[probe] mission-goal bearing body-frame = {math.degrees(goal_bearing_body):.2f} deg (rig: -8.1)")

    meta = {
        "task": TASK, "seed": SEED,
        "cfg": {"decimation": int(env_cfg.decimation), "sim_dt": float(env_cfg.sim.dt),
                "render_interval": int(env_cfg.sim.render_interval), "step_dt": step_dt},
        "terminations_disabled": disabled,
        "anchor_pos_w": anchor_pos, "anchor_yaw_w": anchor_yaw,
        "env_origin": env_origin, "pos_attr": pos_attr,
        "target_world": list(target_world), "target_yaw": START_YAW,
        "ref_world": list(ref_world), "goal_world": list(goal_world),
        "achieved_pos_w": achieved_pos, "achieved_yaw_w": achieved_yaw,
        "quartet_clean_terms": quartet,
        "goal_bearing_body_deg": math.degrees(goal_bearing_body),
        "policy_term_layout": term_layout,
        "settle_steps": SETTLE_STEPS, "spawn_z": SPAWN_Z,
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    # ---- (e) 30 zero-action dump steps ----
    gym_obs_path = os.path.join(OUT_DIR, "gym_obs.jsonl")
    envbuf_path = os.path.join(OUT_DIR, "env_obsbuf.jsonl")
    dumper = make_bridge_obs_dumper(gym_obs_path, "DEPTH_SUBGOAL")
    envbuf_fh = open(envbuf_path, "w")
    envbuf_fh.write(json.dumps({
        "header": True,
        "source": "env ObservationManager 'policy' group row 0",
        "warning": "includes the realism-profile corruption noise the manager injects; "
                   "NOT the clean parity quantity (that is gym_obs.jsonl)",
        "term_layout": term_layout,
    }) + "\n")
    for i in range(DUMP_STEPS):
        obs, _r, _te, _tr, _in = env.step(zero_action.clone())
        t = t_sim_now()
        dumper.write(uw, t)
        row = obs_policy(obs)[0].detach().cpu().numpy().astype(float).tolist()
        envbuf_fh.write(json.dumps({"t_sim": t, "obs": row}) + "\n")
        envbuf_fh.flush()
    dumper.close()
    envbuf_fh.close()
    print(f"[probe] dumped {DUMP_STEPS} gym-obs records -> {gym_obs_path}")

    # ---- closed loop ----
    class ExportedPolicy:
        """play_strafer_navigation.py:205-215 pattern, flat-tensor variant."""

        def __init__(self, path):
            self.model = torch.jit.load(path, map_location=uw.device)
            self.model.eval()
            if hasattr(self.model, "reset"):
                self.model.reset()  # fresh hidden state
            print(f"[probe] loaded exported policy: {path}")

        def __call__(self, flat):
            flat = flat.to(uw.device, dtype=torch.float32)
            if flat.dim() == 1:
                flat = flat.unsqueeze(0)
            with torch.no_grad():
                return self.model(flat)

    def run_closed_loop(policy_path, out_name):
        policy = ExportedPolicy(policy_path)
        o = place_and_settle()
        start_pos, start_yaw = read_pose()
        n_steps = int(round(CLOSED_LOOP_S / step_dt))
        path = os.path.join(OUT_DIR, out_name)
        fh = open(path, "w")
        first_cmd = None
        acts = []
        t0 = t_sim_now()
        for i in range(n_steps):
            flat = obs_policy(o)
            act = policy(flat)
            if first_cmd is None:
                first_cmd = act[0].detach().cpu().tolist()
            o, _r, _te, _tr, _in = env.step(act)
            p, yw = read_pose()
            a = act[0].detach().cpu().tolist()
            acts.append(a)
            fh.write(json.dumps({"i": i, "t_sim": t_sim_now(),
                                 "action": a, "pos_w": p, "yaw_w": yw}) + "\n")
        fh.close()
        end_pos, end_yaw = read_pose()
        disp = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
        net_disp = math.hypot(*disp)
        gvec = (goal_world[0] - start_pos[0], goal_world[1] - start_pos[1])
        gnorm = math.hypot(*gvec)
        toward = (disp[0] * gvec[0] + disp[1] * gvec[1]) / gnorm
        bearing = _wrap(math.atan2(gvec[1], gvec[0]) - start_yaw)
        cmd_dir = math.atan2(first_cmd[1], first_cmd[0])
        off_goal = math.degrees(_wrap(cmd_dir - bearing))
        mean_vx = sum(a[0] for a in acts) / len(acts)
        mean_vy = sum(a[1] for a in acts) / len(acts)
        mean_wz = sum(a[2] for a in acts) / len(acts)
        summary = {
            "policy": policy_path, "trajectory": path,
            "steps": n_steps, "duration_s_sim": n_steps * step_dt,
            "t_sim_span": [t0, t_sim_now()],
            "start_pos_w": start_pos, "start_yaw_w": start_yaw,
            "end_pos_w": end_pos, "end_yaw_w": end_yaw,
            "first_command": first_cmd,
            "first_command_dir_deg": math.degrees(cmd_dir),
            "goal_bearing_body_deg": math.degrees(bearing),
            "first_command_off_goal_deg": off_goal,
            "net_displacement_m": net_disp,
            "net_toward_goal_m": toward,
            "mean_action_vx": mean_vx, "mean_action_vy": mean_vy, "mean_action_wz": mean_wz,
        }
        print(f"[probe] CLOSED-LOOP {out_name}: {json.dumps(summary)}")
        return summary

    cl = {"v2": run_closed_loop(V2_PT, "closed_loop_v2.jsonl")}
    if os.path.exists(V1_PT):
        cl["v1"] = run_closed_loop(V1_PT, "closed_loop_v1.jsonl")
    else:
        cl["v1"] = "v1 .pt not found"
        print("[probe] v1 .pt not found — skipping v1 control")

    with open(os.path.join(OUT_DIR, "closed_loop_summary.json"), "w") as fh:
        json.dump(cl, fh, indent=2)

    print("[probe] DONE")
    EXIT_CODE = 0

except Exception:
    traceback.print_exc()
    EXIT_CODE = 1
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        if env is not None:
            env.close()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    try:
        simulation_app.close()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    sys.stdout.flush()
    # Kit sometimes refuses to exit cleanly (repo conftest precedent) — force it.
    os._exit(EXIT_CODE)
