import mlflow

from .custom_telemetry import CustomTelemetry


class MlFlowCustomTelemetry(CustomTelemetry):
    """MLflow telemetry provider. Enables LiteLLM autologging for token counts and latency."""

    def track(self):
        mlflow.litellm.autolog()
