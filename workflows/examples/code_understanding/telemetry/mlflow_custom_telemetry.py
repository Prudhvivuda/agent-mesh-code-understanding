import os
import mlflow
import litellm
from .custom_telemetry import CustomTelemetry


class MlFlowCustomTelemetry(CustomTelemetry):
    """MLflow telemetry provider. Enables LiteLLM autologging for token counts and latency."""

    def track(self):
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            litellm.callbacks = ["mlflow"]

