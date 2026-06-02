#!/usr/bin/env python3
"""Headless characterization harness for the high-yaw-rate roller bounce.

Spins the Strafer in place on a clean ground plane at a sweep of wheel
speeds and logs chassis vertical position at the PHYSICS substep rate
(120 Hz), high enough to resolve the ~50 Hz roller-passing band that the
30 Hz policy loop aliases away.

For each wheel speed it reports the steady-state chassis-z peak-to-peak,
the growth ratio (late window vs early window), and the dominant vertical
oscillation frequency (FFT). The contact-skip hypothesis predicts the
dominant frequency tracks the roller-passing frequency
(rollers_per_wheel x wheel_rev_rate = 10 x omega/2pi) and scales
linearly with wheel speed.

Optional --contact-offset / --rest-offset apply a PhysX collision-offset
override to all 40 roller covers at load time, so the candidate geometry
fix can be measured without regenerating the USD.

Usage (from repo root, after `source env_setup.sh`):
    $ISAACLAB -p Scripts/roller_bounce_probe.py --headless
    $ISAACLAB -p Scripts/roller_bounce_probe.py --headless \
        --omega-fracs 0.5,0.75,0.9,1.0 --duration 5.0
    $ISAACLAB -p Scripts/roller_bounce_probe.py --headless \
        --omega-fracs 1.0 --contact-offset 0.005 --rest-offset 0.001
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Roller high-omega bounce probe")
    parser.add_argument(
        "--omega-fracs",
        type=str,
        default="0.25,0.5,0.75,1.0",
        help="Comma list of wheel-speed fractions of max (32.67 rad/s) to sweep",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="Seconds of spin to log per omega fraction",
    )
    parser.add_argument(
        "--settle", type=float, default=1.0,
        help="Seconds to let the robot settle on the ground before spinning",
    )
    parser.add_argument(
        "--contact-offset", type=float, default=-1.0,
        help="If >=0, override PhysX contactOffset (m) on all roller covers",
    )
    parser.add_argument(
        "--rest-offset", type=float, default=-1.0,
        help="If >=0, override PhysX restOffset (m) on all roller covers",
    )
    parser.add_argument(
        "--roller-damping", type=float, default=-1.0,
        help="If >=0, override the passive roller_bearings joint damping",
    )
    parser.add_argument(
        "--sim-hz", type=float, default=120.0,
        help="Physics substep rate (default 120 = shared nav sim.dt)",
    )
    parser.add_argument("--solver-type", type=int, default=1,
                        help="PhysX solver: 1=TGS (default), 0=PGS")
    parser.add_argument("--enable-ccd", action="store_true", default=False,
                        help="Continuous collision detection")
    parser.add_argument("--ext-forces-every-iter", action="store_true", default=False,
                        help="Apply external forces every solver iteration")
    parser.add_argument("--solve-art-contact-last", action="store_true", default=False,
                        help="Solve articulation contacts last")
    parser.add_argument(
        "--csv", type=str, default="",
        help="Optional path to write per-substep (omega_frac,t,z,vz) CSV",
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import math
    import numpy as np
    import torch
    import warp as wp

    def to_t(x):
        """Return a torch view whether the data field is torch or a wp.array."""
        if isinstance(x, wp.array):
            return wp.to_torch(x)
        return x

    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext, SimulationCfg
    from isaaclab.assets import Articulation
    from isaaclab_physx.physics import PhysxCfg
    from pxr import UsdPhysics, PhysxSchema

    from strafer_lab.assets import STRAFER_CFG

    MAX_WHEEL_RAD_S = 32.67
    SIM_DT = 1.0 / args.sim_hz

    omega_fracs = [float(x) for x in args.omega_fracs.split(",") if x.strip()]

    # --- Build a minimal scene: ground plane + the shared Strafer cfg ---
    physx_kwargs = dict(
        enable_stabilization=True,  # faithful to the shared nav runtime
        solver_type=args.solver_type,
        enable_ccd=args.enable_ccd,
        enable_external_forces_every_iteration=args.ext_forces_every_iter,
        solve_articulation_contact_last=args.solve_art_contact_last,
    )
    print(f"[probe] physx {physx_kwargs}")
    sim_cfg = SimulationCfg(
        dt=SIM_DT,
        device=args.device if hasattr(args, "device") else "cuda:0",
        physics=PhysxCfg(**physx_kwargs),
    )
    sim = SimulationContext(sim_cfg)

    # Ground plane (restitution 0, like the real ground in the asset bake).
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)

    # Light so headless render (if any) doesn't error; cheap.
    dome = sim_utils.DomeLightCfg(intensity=1000.0)
    dome.func("/World/Light", dome)

    robot_cfg = STRAFER_CFG.replace(prim_path="/World/strafer")
    if args.roller_damping >= 0.0:
        # Copy the actuator cfg so the shared STRAFER_CFG is not mutated.
        import copy
        robot_cfg.actuators = copy.deepcopy(robot_cfg.actuators)
        robot_cfg.actuators["roller_bearings"].damping = args.roller_damping
        print(f"[probe] roller_bearings damping override -> {args.roller_damping}")
    robot = Articulation(robot_cfg)

    # --- Optional collision-offset override on all roller covers ---
    if args.contact_offset >= 0.0 or args.rest_offset >= 0.0:
        stage = sim.stage
        n = 0
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if "roller_cover" not in path:
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            if args.contact_offset >= 0.0:
                api.CreateContactOffsetAttr().Set(float(args.contact_offset))
            if args.rest_offset >= 0.0:
                api.CreateRestOffsetAttr().Set(float(args.rest_offset))
            n += 1
        print(f"[probe] applied collision-offset override to {n} roller covers "
              f"(contact_offset={args.contact_offset}, rest_offset={args.rest_offset})")

    sim.reset()
    # Body world poses are only populated after a step; take one before
    # reading wheel-core positions for left/right detection.
    robot.write_data_to_sim()
    sim.step()
    robot.update(SIM_DT)

    # --- Identify the 4 drive joints and their left/right side ---
    drive_names = [n for n in robot.joint_names if n.endswith("_drive")]
    drive_ids, _ = robot.find_joints([f"{n}" for n in drive_names])
    roller_ids, _ = robot.find_joints(["wheel_[1-4]_roller_[0-9]"])
    # Map each wheel_N_drive to its wheel core body y (base frame) to get side.
    body_names = robot.body_names
    root_pos = to_t(robot.data.root_pos_w)[0]
    body_pos = to_t(robot.data.body_pos_w)[0]  # (num_bodies, 3)
    # Left/right split from the USD-confirmed wheel layout (probe of the
    # physics USD): wheel_1,wheel_3 are +y (left); wheel_2,wheel_4 are -y
    # (right). Pure yaw drives the two sides in opposite directions.
    side_by_wheel = {"wheel_1": 1.0, "wheel_3": 1.0, "wheel_2": -1.0, "wheel_4": -1.0}
    side_sign = torch.tensor(
        [side_by_wheel[jn.replace("_drive", "")] for jn in drive_names],
        device=sim.device,
    )
    print(f"[probe] drive joints {drive_names}")
    print(f"[probe] side signs {side_sign.tolist()}")

    csv_rows = []
    summary = []

    settle_steps = int(args.settle / SIM_DT)
    spin_steps = int(args.duration / SIM_DT)

    for frac in omega_fracs:
        # Reset to spawn state.
        root_state = to_t(robot.data.default_root_state).clone()
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        jp = to_t(robot.data.default_joint_pos).clone()
        jv = jp * 0.0
        robot.write_joint_state_to_sim(jp, jv)
        robot.reset()

        target_w = MAX_WHEEL_RAD_S * frac
        # Pure yaw: left wheels one way, right wheels the other.
        vel_target = (side_sign * target_w).unsqueeze(0)  # (1, 4)

        # Settle (no spin command).
        zero = torch.zeros_like(vel_target)
        for _ in range(settle_steps):
            robot.set_joint_velocity_target(zero, joint_ids=drive_ids)
            robot.write_data_to_sim()
            sim.step()
            robot.update(SIM_DT)

        z0 = float(to_t(robot.data.root_pos_w)[0, 2])

        zs = np.empty(spin_steps, dtype=np.float64)
        vzs = np.empty(spin_steps, dtype=np.float64)
        wz = np.empty(spin_steps, dtype=np.float64)
        drive_w = np.empty(spin_steps, dtype=np.float64)
        roller_w = np.empty(spin_steps, dtype=np.float64)
        for i in range(spin_steps):
            robot.set_joint_velocity_target(vel_target, joint_ids=drive_ids)
            robot.write_data_to_sim()
            sim.step()
            robot.update(SIM_DT)
            zs[i] = float(to_t(robot.data.root_pos_w)[0, 2])
            vzs[i] = float(to_t(robot.data.root_lin_vel_w)[0, 2])
            wz[i] = float(to_t(robot.data.root_ang_vel_w)[0, 2])
            jvel = to_t(robot.data.joint_vel)[0]
            drive_w[i] = float(jvel[drive_ids].abs().mean())
            roller_w[i] = float(jvel[roller_ids].abs().max())
            if args.csv:
                csv_rows.append((frac, i * SIM_DT, zs[i], vzs[i]))

        # Metrics. Use the second half as "late", first quarter as "early".
        q = spin_steps // 4
        h = spin_steps // 2
        early_p2p = float(zs[:q].max() - zs[:q].min())
        late_p2p = float(zs[h:].max() - zs[h:].min())
        growth = late_p2p / early_p2p if early_p2p > 1e-9 else float("inf")
        # FFT of detrended z over the spin window; report top-3 peaks >=3 Hz.
        def top_peaks(sig, k=3):
            sd = sig - sig.mean()
            freqs = np.fft.rfftfreq(len(sig), d=SIM_DT)
            mag = np.abs(np.fft.rfft(sd))
            band = freqs >= 3.0
            fb, mb = freqs[band], mag[band]
            idx = np.argsort(mb)[::-1][:k]
            return [(float(fb[j]), float(mb[j])) for j in idx]

        z_peaks = top_peaks(zs)
        vz_peaks = top_peaks(vzs)
        peak_f = z_peaks[0][0] if z_peaks else 0.0
        wheel_rev = target_w / (2 * math.pi)
        roller_pass_f = 10.0 * wheel_rev
        # Achieved (measured) speeds, averaged over the steady (late) half.
        ach_drive = float(np.mean(drive_w[h:]))
        ach_roller = float(np.mean(roller_w[h:]))
        ach_rev = ach_drive / (2 * math.pi)
        ach_roller_pass = 10.0 * ach_rev
        mean_wz = float(np.mean(np.abs(wz[h:])))
        summary.append(dict(
            frac=frac, wheel_w=target_w, wheel_rev=wheel_rev,
            roller_pass_f=roller_pass_f, early_p2p=early_p2p,
            late_p2p=late_p2p, growth=growth, peak_f=peak_f,
            max_z=float(zs.max()), z0=z0, mean_wz=mean_wz,
            ach_drive=ach_drive, ach_roller=ach_roller,
            ach_roller_pass=ach_roller_pass, z_peaks=z_peaks, vz_peaks=vz_peaks,
        ))
        zp = " ".join(f"{f:.1f}Hz" for f, _ in z_peaks)
        print(f"[probe] frac={frac:.2f} cmd_w={target_w:5.2f} ach_w={ach_drive:5.2f} "
              f"roller={ach_roller:5.1f} rad/s | cmd-rollpass {roller_pass_f:4.1f} "
              f"ach-rollpass {ach_roller_pass:4.1f} Hz | z p2p early {early_p2p*1000:6.2f} "
              f"late {late_p2p*1000:6.2f}mm growth {growth:4.2f}x | z-peaks [{zp}] | "
              f"max_z {zs.max()*1000:5.1f}mm | yaw {mean_wz:4.1f} rad/s")

    print("\n========== SUMMARY ==========")
    print(f"{'frac':>5} {'ach_w':>6} {'rollHz':>7} {'bounceHz':>9} "
          f"{'b/roll':>6} {'b/rev':>6} {'late_p2p_mm':>11} {'growth':>7} {'yaw':>6}")
    for s in summary:
        rp = s['ach_roller_pass']
        rev = rp / 10.0
        ratio = s['peak_f'] / rp if rp > 0 else 0.0
        ratio_rev = s['peak_f'] / rev if rev > 0 else 0.0
        print(f"{s['frac']:>5.2f} {s['ach_drive']:>6.2f} {rp:>7.1f} "
              f"{s['peak_f']:>9.1f} {ratio:>6.2f} {ratio_rev:>6.2f} "
              f"{s['late_p2p']*1000:>11.2f} {s['growth']:>7.2f} {s['mean_wz']:>6.1f}")
    print("Interpretation: if bounceHz ~= rollHz (or a fixed integer ratio) and")
    print("scales with w, while p2p/growth rise sharply only near frac=1.0, the")
    print("discrete-roller contact-skip hypothesis is confirmed.")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("omega_frac,t,z,vz\n")
            for r in csv_rows:
                f.write(f"{r[0]},{r[1]:.6f},{r[2]:.6f},{r[3]:.6f}\n")
        print(f"[probe] wrote {len(csv_rows)} rows to {args.csv}")

    simulation_app.close()


if __name__ == "__main__":
    main()
