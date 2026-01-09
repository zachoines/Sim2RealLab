"""
Bring up Isaac Lab with an empty stage using the installed pip build of Isaac Sim.

Usage:
    ./launch_isaac_lab.ps1
    # or forward any AppLauncher flags, e.g.:
    ./launch_isaac_lab.ps1 -- --/renderer/multiGpu/enabled=false
"""

import argparse

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch an empty Isaac Lab stage.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Spin up the Omniverse app (headed unless --headless is passed through).
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import sim modules only after SimulationApp is instantiated to ensure omni.* is available.
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim import PhysxCfg

    # Increase GPU buffer for collision pairs (needed for robots with many collision shapes)
    physx_cfg = PhysxCfg(
        gpu_found_lost_aggregate_pairs_capacity=2**18,  # 262144, default is much smaller
        gpu_total_aggregate_pairs_capacity=2**18,
    )

    # Minimal simulation context with default physics/render rates.
    sim_cfg = SimulationCfg(physx=physx_cfg)
    sim = SimulationContext(sim_cfg)

    # Main loop: nothing spawned, just an empty stage and active viewport.
    while simulation_app.is_running():
        sim.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
