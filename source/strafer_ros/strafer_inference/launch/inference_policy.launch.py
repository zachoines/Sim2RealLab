"""Policy-backend launch for the containerized inference service (strafer-gpu).

The `inference` compose service (`docker compose --profile policy`) runs this
instead of `inference.launch.py` so the canonical `autonomy.launch.py` backend
coupling is reproduced in the decomposed deployment, where inference/subgoal run
in their own container discovered over DDS:

  - policy backend (``strafer_direct`` | ``hybrid_nav2_strafer``) -> the
    ``strafer_inference`` node (via ``inference.launch.py``);
  - ``hybrid_nav2_strafer`` -> ALSO the rolling-subgoal generator
    (``subgoal_generator.launch.py``); without it ``/strafer/subgoal`` never
    publishes, the inference watchdog's subgoal source never goes fresh, and
    every hybrid mission zero-twists. hybrid is the only backend the depth
    policy runs under.
  - a policy backend with an empty/missing model -> **HARD ERROR at startup**
    (fail loud). The canonical design refuses to advertise ``navigate_to_pose``
    and would silently fall back to nav2; in the policy profile that silent
    fallback is a footgun, so this launch aborts instead.

Reads ``STRAFER_NAV_BACKEND`` / ``STRAFER_INFERENCE_MODEL_PATH`` /
``STRAFER_POLICY_VARIANT`` / ``STRAFER_USE_SIM_TIME`` from the environment
(compose ``autonomy.env``).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

# navigate_to_pose backends that run the RL inference node; the subgoal
# generator is needed only for the hybrid one (mirrors autonomy.launch.py).
_POLICY_BACKENDS = {"strafer_direct", "hybrid_nav2_strafer"}


def generate_launch_description() -> LaunchDescription:
    inference_dir = get_package_share_directory("strafer_inference")

    backend = os.environ.get("STRAFER_NAV_BACKEND", "nav2")
    model = os.environ.get("STRAFER_INFERENCE_MODEL_PATH", "").strip()
    variant = os.environ.get("STRAFER_POLICY_VARIANT", "DEPTH")
    use_sim_time = os.environ.get("STRAFER_USE_SIM_TIME", "false")

    # Fail loud #1 — the inference container belongs to the `policy` profile;
    # it is meaningless under nav2.
    if backend not in _POLICY_BACKENDS:
        raise RuntimeError(
            f"strafer-gpu inference service started but STRAFER_NAV_BACKEND={backend!r} "
            f"is not a policy backend ({sorted(_POLICY_BACKENDS)}). The inference "
            f"container is the `policy` compose profile; run it only with "
            f"STRAFER_NAV_BACKEND=strafer_direct or hybrid_nav2_strafer."
        )

    # Fail loud #2 — a policy backend with no model is a misconfiguration, not a
    # silent nav2 fallback.
    if not model or not os.path.isfile(model):
        raise RuntimeError(
            f"STRAFER_NAV_BACKEND={backend} but STRAFER_INFERENCE_MODEL_PATH is "
            f"empty or not a file ({model!r}). Mount the exported policy under "
            f"/models and set STRAFER_INFERENCE_MODEL_PATH. Aborting rather than "
            f"advertise no navigate_to_pose and silently degrade to nav2."
        )

    entities = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(inference_dir, "launch", "inference.launch.py")
            ),
            launch_arguments={
                "model_path": model,
                "policy_variant": variant,
                "use_sim_time": use_sim_time,
            }.items(),
        ),
    ]

    if backend == "hybrid_nav2_strafer":
        entities.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(inference_dir, "launch", "subgoal_generator.launch.py")
            ),
            launch_arguments={"use_sim_time": use_sim_time}.items(),
        ))

    return LaunchDescription(entities)
