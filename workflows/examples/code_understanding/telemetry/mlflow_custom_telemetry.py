import logging
import os
import mlflow
import litellm
from .custom_telemetry import CustomTelemetry

logger = logging.getLogger(__name__)


class MlFlowCustomTelemetry(CustomTelemetry):
    """MLflow telemetry provider. Enables LiteLLM autologging for token counts and latency."""

    def track(self):
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        logger.info(f"MlFlowCustomTelemetry.track() called. MLFLOW_TRACKING_URI={tracking_uri}")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        try:
            mlflow.openai.autolog()
            logger.info("mlflow.openai.autolog() registered successfully")
        except Exception as e:
            logger.error(f"mlflow.openai.autolog() failed: {e}")
        litellm.callbacks = ["mlflow"]
        logger.info("litellm.callbacks set to ['mlflow']")

