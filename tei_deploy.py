#!/usr/bin/env python3
"""Create an OCI Data Science deployment for a TEI model in OCIR.

The script reads the deployment settings from ``config.yaml`` by default.  Pass
the registered model OCID with ``--model-id`` (or set ``model_id`` in the YAML).
The project OCID is read from the model when it is not explicitly configured.

Required YAML keys: compartment_id, deployment_display_name, deployment_shape,
container_image.  For a custom network, also set subnet_id or
private_endpoint_id.  ``commands`` and ``entry_point`` can be strings or YAML
lists.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

try:
    import oci
    from oci.data_science import DataScienceClient
    from oci.data_science import models as data_science_models
    from oci.data_science.models import (
        CreateModelDeploymentDetails,
        FixedSizeScalingPolicy,
        InstanceConfiguration,
        ModelConfigurationDetails,
        OcirModelDeploymentEnvironmentConfigurationDetails,
        SingleModelDeploymentConfigurationDetails,
    )
except ImportError as error:
    raise SystemExit(
        "The OCI Python SDK is required. Install it with: python3 -m pip install oci"
    ) from error


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config


def required(config: dict[str, Any], name: str) -> Any:
    value = config.get(name)
    if value in (None, "", f"<{name.replace('_', '-')}>"):
        raise ValueError(f"Missing required configuration value: {name}")
    return value


def optional_string_list(config: dict[str, Any], name: str) -> list[str] | None:
    value = config.get(name)
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a string or a list of strings")
    return value


def embed_endpoint() -> Any:
    """Return the OCI custom endpoint definition for TEI's POST /embed API."""
    supported_environment_fields = (
        OcirModelDeploymentEnvironmentConfigurationDetails().swagger_types
    )
    endpoint_class = getattr(data_science_models, "InferenceHttpEndpoint", None)
    if not endpoint_class or "custom_http_endpoints" not in supported_environment_fields:
        raise RuntimeError(
            "The installed OCI Python SDK does not support custom model-deployment "
            "endpoints. Upgrade it before deploying: python3 -m pip install --upgrade oci"
        )
    http_method = getattr(getattr(data_science_models, "HttpMethod", None), "POST", "POST")
    return endpoint_class(endpoint_uri_suffix="/embed", http_methods=[http_method])


def model_project_id(client: DataScienceClient, model_id: str, config: dict[str, Any]) -> str:
    """Use the configured project, or discover it from the registered model."""
    project_id = config.get("project_id")
    if project_id and project_id not in ("None", "<project-ocid>"):
        return project_id
    project_id = client.get_model(model_id).data.project_id
    if not project_id:
        raise ValueError(f"Model {model_id} does not have an associated project_id")
    return project_id


def build_deployment_details(
    client: DataScienceClient, model_id: str, config: dict[str, Any]
) -> CreateModelDeploymentDetails:
    """Build the complete payload expected by CreateModelDeploymentDetails."""
    compartment_id = required(config, "compartment_id")
    project_id = model_project_id(client, model_id, config)

    instance_kwargs: dict[str, Any] = {
        "instance_shape_name": required(config, "deployment_shape"),
    }
    # network_access_type was added to the OCI SDK after the original model
    # deployment API. Send it only when this installed SDK supports it.
    network_access_type = config.get("network_access_type")
    supported_instance_fields = InstanceConfiguration().swagger_types
    if network_access_type and "network_access_type" in supported_instance_fields:
        instance_kwargs["network_access_type"] = network_access_type
    elif network_access_type:
        LOG.warning(
            "Ignoring network_access_type because this OCI SDK does not support it; "
            "upgrade the oci package to use that setting."
        )
    # OCI accepts either a Data Science private endpoint or a subnet, depending
    # on the selected network access type. Do not send blank optional fields.
    for key in ("subnet_id", "private_endpoint_id"):
        if config.get(key):
            instance_kwargs[key] = config[key]

    instance_configuration = InstanceConfiguration(**instance_kwargs)
    model_configuration = ModelConfigurationDetails(
        model_id=model_id,
        instance_configuration=instance_configuration,
        # Explicitly pin one A10 instance instead of relying on an implicit
        # service default. This is the scaling policy missing from the old call.
        scaling_policy=FixedSizeScalingPolicy(
            policy_type="FIXED_SIZE",
            instance_count=int(config.get("instance_count", 1)),
        ),
    )

    environment_kwargs: dict[str, Any] = {
        "environment_configuration_type": "OCIR_CONTAINER",
        "image": required(config, "container_image"),
        "server_port": int(config.get("server_port", 8080)),
        "health_check_port": int(config.get("health_check_port", 8080)),
    }
    entrypoint = optional_string_list(config, "entry_point")
    commands = optional_string_list(config, "commands")
    if entrypoint:
        environment_kwargs["entrypoint"] = entrypoint
    if commands:
        environment_kwargs["cmd"] = commands
    if config.get("environment_variables"):
        environment_kwargs["environment_variables"] = config["environment_variables"]
    # OCI exposes this as <model-deployment-url>/predict/embed and strips the
    # /predict prefix before forwarding to TEI's /embed container route.
    environment_kwargs["custom_http_endpoints"] = [embed_endpoint()]

    deployment_configuration = SingleModelDeploymentConfigurationDetails(
        deployment_type="SINGLE_MODEL",
        model_configuration_details=model_configuration,
        environment_configuration_details=OcirModelDeploymentEnvironmentConfigurationDetails(
            **environment_kwargs
        ),
    )
    return CreateModelDeploymentDetails(
        compartment_id=compartment_id,
        project_id=project_id,
        display_name=required(config, "deployment_display_name"),
        description=config.get("deployment_description", ""),
        model_deployment_configuration_details=deployment_configuration,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--model-id", help="Registered OCI Data Science model OCID")
    parser.add_argument("--profile", default="DEFAULT", help="OCI CLI config profile")
    parser.add_argument("--wait", action="store_true", help="Wait until deployment becomes ACTIVE")
    args = parser.parse_args()

    config = load_config(args.config)
    model_id = args.model_id or config.get("model_id")
    if not model_id:
        parser.error("provide --model-id or set model_id in the YAML config")

    client = DataScienceClient(oci.config.from_file(profile_name=args.profile), timeout=(60, 1800))
    details = build_deployment_details(client, model_id, config)
    response = client.create_model_deployment(details)
    deployment = response.data
    print(f"Deployment created: {deployment.id}")
    print(f"Lifecycle state: {deployment.lifecycle_state}")

    if args.wait:
        deployment = oci.wait_until(
            client,
            client.get_model_deployment(deployment.id),
            evaluate_response=lambda r: r.data.lifecycle_state in ("ACTIVE", "FAILED"),
            max_wait_seconds=3600,
        ).data
        print(f"Final lifecycle state: {deployment.lifecycle_state}")
        if deployment.lifecycle_state != "ACTIVE":
            raise RuntimeError(deployment.lifecycle_details or "Model deployment failed")
        print(f"Deployment URL: {deployment.model_deployment_url}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
