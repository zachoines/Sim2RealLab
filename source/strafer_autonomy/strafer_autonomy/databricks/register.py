"""Log the planner and VLM pyfunc models to MLflow / Databricks.

Usage:

    python -m strafer_autonomy.databricks.register \\
        --planner-model-path /path/to/Qwen3-4B \\
        --vlm-model-path /path/to/Qwen2.5-VL-3B-Instruct \\
        --experiment /Shared/strafer \\
        --register-name-planner strafer-planner \\
        --register-name-vlm strafer-vlm

Requires ``mlflow`` installed in the active environment. Heavy imports
(mlflow, torch, transformers) are done lazily inside main() so this
module itself remains importable on the Jetson.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


_PLANNER_PIP_REQUIREMENTS = (
    "transformers>=4.44",
    "torch>=2.4",
    "accelerate>=0.33",
    "pydantic>=2.0",
)

_VLM_PIP_REQUIREMENTS = (
    "transformers>=4.44",
    "torch>=2.4",
    "accelerate>=0.33",
    "qwen-vl-utils",
    "Pillow>=10.0",
    "pydantic>=2.0",
)


@dataclass(frozen=True)
class RegistrationArgs:
    planner_model_path: str
    vlm_model_path: str
    experiment: str
    register_name_planner: str | None
    register_name_vlm: str | None
    artifact_path_planner: str
    artifact_path_vlm: str
    log_planner: bool
    log_vlm: bool


def parse_args(argv: list[str] | None = None) -> RegistrationArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-model-path", required=True)
    parser.add_argument("--vlm-model-path", required=True)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--register-name-planner", default=None)
    parser.add_argument("--register-name-vlm", default=None)
    parser.add_argument("--artifact-path-planner", default="strafer_planner")
    parser.add_argument("--artifact-path-vlm", default="strafer_vlm")
    parser.add_argument("--skip-planner", action="store_true")
    parser.add_argument("--skip-vlm", action="store_true")
    args = parser.parse_args(argv)
    return RegistrationArgs(
        planner_model_path=args.planner_model_path,
        vlm_model_path=args.vlm_model_path,
        experiment=args.experiment or "/Shared/strafer",
        register_name_planner=args.register_name_planner,
        register_name_vlm=args.register_name_vlm,
        artifact_path_planner=args.artifact_path_planner,
        artifact_path_vlm=args.artifact_path_vlm,
        log_planner=not args.skip_planner,
        log_vlm=not args.skip_vlm,
    )


def _build_planner_pyfunc():
    """Return a subclass that mixes ``StraferPlannerModel`` with ``PythonModel``.

    Keeping this behind a function lets us defer the ``mlflow`` import so
    the module is still importable on machines without MLflow installed.
    """
    import mlflow

    from strafer_autonomy.databricks.planner_model import StraferPlannerModel

    class _MlflowStraferPlannerModel(StraferPlannerModel, mlflow.pyfunc.PythonModel):  # type: ignore[misc]
        pass

    return _MlflowStraferPlannerModel


def _build_vlm_pyfunc():
    import mlflow

    from strafer_autonomy.databricks.vlm_model import StraferVLMModel

    class _MlflowStraferVLMModel(StraferVLMModel, mlflow.pyfunc.PythonModel):  # type: ignore[misc]
        pass

    return _MlflowStraferVLMModel


def log_planner(args: RegistrationArgs) -> str:
    import mlflow

    PlannerCls = _build_planner_pyfunc()
    logger.info("Logging planner model from %s", args.planner_model_path)
    with mlflow.start_run(run_name="strafer-planner-log") as run:
        mlflow.pyfunc.log_model(
            artifact_path=args.artifact_path_planner,
            python_model=PlannerCls(),
            artifacts={"model": args.planner_model_path},
            pip_requirements=list(_PLANNER_PIP_REQUIREMENTS),
            registered_model_name=args.register_name_planner,
        )
    return run.info.run_id


def log_vlm(args: RegistrationArgs) -> str:
    import mlflow

    VLMCls = _build_vlm_pyfunc()
    logger.info("Logging VLM model from %s", args.vlm_model_path)
    with mlflow.start_run(run_name="strafer-vlm-log") as run:
        mlflow.pyfunc.log_model(
            artifact_path=args.artifact_path_vlm,
            python_model=VLMCls(),
            artifacts={"model": args.vlm_model_path},
            pip_requirements=list(_VLM_PIP_REQUIREMENTS),
            registered_model_name=args.register_name_vlm,
        )
    return run.info.run_id


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    try:
        import mlflow  # noqa: F401
    except ImportError:
        logger.error(
            "mlflow is not installed. Install with `pip install mlflow` before "
            "running register.py."
        )
        return 2

    import mlflow

    mlflow.set_experiment(args.experiment)

    if args.log_planner:
        run_id = log_planner(args)
        logger.info("Planner logged in run %s", run_id)
    if args.log_vlm:
        run_id = log_vlm(args)
        logger.info("VLM logged in run %s", run_id)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
