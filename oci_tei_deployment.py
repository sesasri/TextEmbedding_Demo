#!/usr/bin/env python3
"""Stage a TEI model ZIP in OCI Object Storage, register it, and deploy it.

The ZIP is uploaded from ``zip_file_path`` to Object Storage first.  The same
object is then streamed into the Data Science Model Catalog as the immutable
model artifact.  Set ``deployment_display_name`` to an empty value, or pass
``--skip-deploy``, to only create the catalog model and artifact.
"""

import argparse
import logging
import os

import oci
import yaml
from oci.data_science import DataScienceClient
from oci.data_science.models import CreateModelDetails, CreateModelProvenanceDetails
from oci.object_storage import ObjectStorageClient

from oci_tei_deployment import create_model_deployment, create_project


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ObjectStorageArtifactStream:
    """Expose an Object Storage response as a read-only upload stream.

    The OCI SDK's bundled Requests library checks ``fileno()`` on a raw
    urllib3 response and then expects a local-file ``mode`` attribute.  This
    minimal adapter intentionally exposes only ``read()`` so the supplied
    ``content_length`` header is used instead.
    """

    def __init__(self, raw_response, content_length):
        self._raw_response = raw_response
        # OCI's SDK checks this standard Requests attribute when determining
        # whether it can send a stream without first buffering the body.
        self.len = content_length

    def read(self, size=-1):
        return self._raw_response.read(size)


def load_config(path):
    """Load YAML configuration from *path*."""
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def required(config, key):
    """Return a required configuration value and reject documentation placeholders."""
    value = config.get(key)
    if not value or (isinstance(value, str) and value.startswith("<")):
        raise ValueError(f"{key} must be set in the configuration file")
    return value


def project_id_for_config(data_science_client, config):
    """Return the configured project ID, or create the configured project."""
    project_id = config.get("project_id")
    if project_id and project_id not in {"None", "<project-ocid>"}:
        return project_id

    return create_project(
        data_science_client,
        required(config, "compartment_id"),
        required(config, "project_display_name"),
        config.get("project_description", ""),
    )


def create_catalog_model(data_science_client, config, project_id):
    """Create the Model Catalog record and optional provenance metadata."""
    details = {
        "compartment_id": required(config, "compartment_id"),
        "project_id": project_id,
        "display_name": required(config, "display_name"),
        "description": config.get("description", ""),
    }
    for config_key, sdk_key in (
        ("custom_metadata_list", "custom_metadata_list"),
        ("defined_metadata_list", "defined_metadata_list"),
        ("input_schema_str", "input_schema"),
        ("output_schema_str", "output_schema"),
    ):
        if config.get(config_key):
            details[sdk_key] = config[config_key]

    logger.info("Creating Model Catalog record: %s", details["display_name"])
    model = data_science_client.create_model(CreateModelDetails(**details)).data

    provenance = config.get("provenance") or {}
    provenance_details = CreateModelProvenanceDetails(
        repository_url=provenance.get("repository_url", ""),
        git_branch=provenance.get("git_branch", ""),
        git_commit=provenance.get("git_commit", ""),
        script_dir=provenance.get("script_dir", ""),
        training_script=provenance.get("training_script", ""),
        training_id=provenance.get("training_id", ""),
    )
    if any(
        (
            provenance_details.repository_url,
            provenance_details.git_branch,
            provenance_details.git_commit,
            provenance_details.training_script,
            provenance_details.training_id,
        )
    ):
        data_science_client.create_model_provenance(model.id, provenance_details)

    return model.id


def object_storage_settings(object_storage_client, config):
    """Resolve Object Storage namespace, bucket, and destination object name."""
    storage = config.get("object_storage") or {}
    namespace = storage.get("namespace") or object_storage_client.get_namespace().data
    bucket_name = required(storage, "bucket_name")
    object_name = storage.get("object_name")
    if not object_name:
        object_name = os.path.basename(required(config, "zip_file_path"))
    return namespace, bucket_name, object_name


def upload_zip_to_object_storage(object_storage_client, config):
    """Upload the local ZIP without reading its contents into memory."""
    zip_path = required(config, "zip_file_path")
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Model ZIP not found: {zip_path}")

    namespace, bucket_name, object_name = object_storage_settings(
        object_storage_client, config
    )
    content_length = os.path.getsize(zip_path)
    logger.info(
        "Uploading %s (%d bytes) to oci://%s@%s/%s",
        zip_path,
        content_length,
        bucket_name,
        namespace,
        object_name,
    )
    with open(zip_path, "rb") as zip_stream:
        object_storage_client.put_object(
            namespace_name=namespace,
            bucket_name=bucket_name,
            object_name=object_name,
            put_object_body=zip_stream,
            content_length=content_length,
            content_type="application/zip",
        )
    return namespace, bucket_name, object_name


def create_artifact_from_object_storage(
    data_science_client, object_storage_client, model_id, namespace, bucket_name, object_name
):
    """Stream the staged Object Storage ZIP directly into Model Catalog.

    This upload has no retry strategy because an HTTP response body cannot be
    rewound after a failed request.  Re-running the script fetches a new stream
    from Object Storage.
    """
    logger.info("Creating model artifact from oci://%s@%s/%s", bucket_name, namespace, object_name)
    response = object_storage_client.get_object(
        namespace_name=namespace,
        bucket_name=bucket_name,
        object_name=object_name,
    )
    content_length = int(response.headers["content-length"])
    try:
        data_science_client.create_model_artifact(
            model_id=model_id,
            model_artifact=ObjectStorageArtifactStream(response.data.raw, content_length),
            content_length=content_length,
            content_disposition=f'attachment; filename="{os.path.basename(object_name)}"',
            retry_strategy=oci.retry.NoneRetryStrategy(),
        )
    finally:
        response.data.raw.close()


def run(config_path, skip_deploy=False):
    """Upload, register, and optionally deploy the configured TEI model."""
    config = load_config(config_path)
    oci_config = oci.config.from_file()
    data_science_client = DataScienceClient(oci_config, timeout=(60, 1800))
    object_storage_client = ObjectStorageClient(oci_config, timeout=(60, 1800))

    project_id = project_id_for_config(data_science_client, config)
    namespace, bucket_name, object_name = upload_zip_to_object_storage(
        object_storage_client, config
    )
    model_id = create_catalog_model(data_science_client, config, project_id)
    create_artifact_from_object_storage(
        data_science_client,
        object_storage_client,
        model_id,
        namespace,
        bucket_name,
        object_name,
    )
    logger.info("Model artifact created successfully. Model OCID: %s", model_id)

    deployment_name = config.get("deployment_display_name")
    if not skip_deploy and deployment_name and not deployment_name.startswith("<"):
        deployment_id = create_model_deployment(
            data_science_client,
            model_id,
            project_id,
            required(config, "compartment_id"),
            deployment_name,
            config.get("deployment_description", ""),
            config,
        )
        logger.info("Deployment OCID: %s", deployment_id)
    return model_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="YAML configuration path")
    parser.add_argument(
        "--skip-deploy", action="store_true", help="Create the model artifact only"
    )
    args = parser.parse_args()
    run(args.config, skip_deploy=args.skip_deploy)


if __name__ == "__main__":
    main()
