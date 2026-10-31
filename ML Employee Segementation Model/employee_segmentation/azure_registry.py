"""Section 10: save the trained pipeline and register it in the Azure ML Model Registry."""
import pickle

from . import config


def save_pipeline(pipeline, path=config.MODEL_PATH):
    """Save the pipeline to a local file (needed so Azure ML can upload it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print("Pipeline saved for upload:", path)
    return path


def load_pipeline(path=config.MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def register_model(path=config.MODEL_PATH):
    """Upload to Azure ML; the same name creates the next version automatically."""
    from azure.ai.ml import MLClient
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Model
    from azure.identity import DefaultAzureCredential

    missing = [n for n, v in [("AZURE_SUBSCRIPTION_ID", config.AZURE_SUBSCRIPTION_ID),
                              ("AZURE_RESOURCE_GROUP", config.AZURE_RESOURCE_GROUP),
                              ("AZURE_WORKSPACE_NAME", config.AZURE_WORKSPACE_NAME)] if not v]
    if missing:
        raise EnvironmentError(f"Set these environment variables first: {', '.join(missing)}")

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=config.AZURE_SUBSCRIPTION_ID,
        resource_group_name=config.AZURE_RESOURCE_GROUP,
        workspace_name=config.AZURE_WORKSPACE_NAME,
    )
    model = Model(
        path=str(path),
        name=config.AZURE_MODEL_NAME,
        description=config.AZURE_MODEL_DESCRIPTION,
        type=AssetTypes.CUSTOM_MODEL,
    )
    registered = ml_client.models.create_or_update(model)
    print("Model registered successfully!")
    print(f"   Name    : {registered.name}")
    print(f"   Version : {registered.version}")
    return registered
