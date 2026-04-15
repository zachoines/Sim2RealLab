"""MLflow ``pyfunc`` wrappers for the planner and VLM models.

These modules are used by the Databricks deployment pipeline
(`databricks/register.py`) to log the planner and VLM as MLflow models so
they can be served via Databricks Model Serving endpoints. The wrappers
replicate the full service-side inference pipeline so the serving response
matches what the LAN HTTP services return.

They are intentionally lightweight imports — heavy dependencies
(``mlflow``, ``transformers``, ``torch``) are imported lazily inside the
model classes so ``strafer_autonomy.databricks`` can be imported on the
Jetson without pulling in MLflow.
"""

from strafer_autonomy.databricks.planner_model import StraferPlannerModel
from strafer_autonomy.databricks.vlm_model import StraferVLMModel

__all__ = ["StraferPlannerModel", "StraferVLMModel"]
