import os
import mlflow

from .custom_telemetry import CustomTelemetry


class MlFlowCustomTelemetry(CustomTelemetry):
    """MLflow telemetry provider. Enables LiteLLM autologging for token counts and latency."""

    _EXPERIMENT_NAME = f"{os.environ.get('MLFLOW_NAMESPACE', os.environ.get('KFP_NAMESPACE', 'demo'))}/code-refactoring/traces"

    def track(self):
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self._EXPERIMENT_NAME)
        mlflow.litellm.autolog()
