#!/usr/bin/env python3
"""
OCI Data Science TEI Model Deployment Script
Registers a model artifact and creates an OCI Data Science model deployment.
"""

import os
import yaml
import logging
from oci.data_science import DataScienceClient
from oci.data_science import models as data_science_models
from oci.data_science.models import (
    CreateModelDetails, 
    CreateModelProvenanceDetails, 
    CreateProjectDetails,
    CreateModelDeploymentDetails,
    SingleModelDeploymentConfigurationDetails,
    ModelConfigurationDetails,
    OcirModelDeploymentEnvironmentConfigurationDetails,
    InstanceConfiguration,
    FixedSizeScalingPolicy
)
from oci.config import from_file

# Configuration file path
CONFIG_FILE = 'config.yaml'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_file):
    """Load configuration from YAML file"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_project(data_science_client, compartment_id, display_name, description):
    """Create a new OCI Data Science project"""
    print(f"Creating project: {display_name}")
    project_details = CreateProjectDetails(
        compartment_id=compartment_id,
        display_name=display_name,
        description=description
    )
    project = data_science_client.create_project(project_details)
    project_id = project.data.id
    print(f"Project created with ID: {project_id}")
    return project_id


def embed_endpoint():
    """Return the OCI custom endpoint definition for TEI's POST /embed API."""
    supported_environment_fields = OcirModelDeploymentEnvironmentConfigurationDetails().swagger_types
    endpoint_class = getattr(data_science_models, 'InferenceHttpEndpoint', None)
    if not endpoint_class or 'custom_http_endpoints' not in supported_environment_fields:
        raise RuntimeError(
            'The installed OCI Python SDK does not support custom model-deployment '
            'endpoints. Upgrade it before deploying: python3 -m pip install --upgrade oci'
        )
    http_method = getattr(getattr(data_science_models, 'HttpMethod', None), 'POST', 'POST')
    return endpoint_class(endpoint_uri_suffix='/embed', http_methods=[http_method])


def create_model_deployment(data_science_client, model_id, project_id, compartment_id, display_name, description, config):
    """Create a complete, SDK-compatible OCI Data Science model deployment."""
    print(f"Creating model deployment: {display_name}")

    instance_kwargs = {
        'instance_shape_name': config['deployment_shape'],
    }
    # Older OCI SDK versions do not have network_access_type. Only include the
    # field when the installed SDK supports it, as tei_deploy.py does.
    network_access_type = config.get('network_access_type')
    supported_instance_fields = InstanceConfiguration().swagger_types
    if network_access_type and 'network_access_type' in supported_instance_fields:
        instance_kwargs['network_access_type'] = network_access_type
    elif network_access_type:
        logger.warning(
            "Ignoring network_access_type because this OCI SDK does not support it; "
            "upgrade the oci package to use that setting."
        )
    for key in ('subnet_id', 'private_endpoint_id'):
        if config.get(key):
            instance_kwargs[key] = config[key]

    instance_configuration = InstanceConfiguration(**instance_kwargs)
    model_configuration_details = ModelConfigurationDetails(
        model_id=model_id,
        instance_configuration=instance_configuration,
        scaling_policy=FixedSizeScalingPolicy(
            policy_type='FIXED_SIZE',
            instance_count=int(config.get('instance_count', 1)),
        ),
    )

    entrypoint = config.get('entry_point')
    if isinstance(entrypoint, str):
        entrypoint = [entrypoint]
    commands = config.get('commands')
    if isinstance(commands, str):
        commands = [commands]
    if entrypoint is not None and not isinstance(entrypoint, list):
        raise ValueError('entry_point must be a string or a list of strings')
    if commands is not None and not isinstance(commands, list):
        raise ValueError('commands must be a string or a list of strings')

    environment_kwargs = {
        'environment_configuration_type': 'OCIR_CONTAINER',
        'image': config['container_image'],
        'server_port': int(config.get('server_port', 8080)),
        'health_check_port': int(config.get('health_check_port', 8080)),
    }
    if entrypoint:
        environment_kwargs['entrypoint'] = entrypoint
    if commands:
        environment_kwargs['cmd'] = commands
    if config.get('environment_variables'):
        environment_kwargs['environment_variables'] = config['environment_variables']
    # OCI exposes this as <model-deployment-url>/predict/embed and strips the
    # /predict prefix before forwarding to TEI's /embed container route.
    environment_kwargs['custom_http_endpoints'] = [embed_endpoint()]

    environment_config_details = OcirModelDeploymentEnvironmentConfigurationDetails(**environment_kwargs)
    single_model_deployment_config_details = SingleModelDeploymentConfigurationDetails(
        deployment_type='SINGLE_MODEL',
        model_configuration_details=model_configuration_details,
        environment_configuration_details=environment_config_details,
    )

    deployment_details = CreateModelDeploymentDetails(
        compartment_id=compartment_id,
        project_id=project_id,
        display_name=display_name,
        description=description,
        model_deployment_configuration_details=single_model_deployment_config_details,
    )
    deployment = data_science_client.create_model_deployment(deployment_details)
    deployment_id = deployment.data.id
    print(f"Model deployment created with ID: {deployment_id}")
    return deployment_id


def upload_and_deploy_model():
    """Register a model artifact and optionally create its deployment."""
    
    # Load configuration
    config = load_config(CONFIG_FILE)
    
    # Initialize OCI client
    oci_config = from_file()
    data_science_client = DataScienceClient(oci_config,timeout=(60, 1800))
    
    # Create project if PROJECT_ID is not provided
    project_id = config.get('project_id')
    # Handle string 'None' or empty values
    if not project_id or project_id == '<project-ocid>' or project_id == 'None':
        project_display_name = config.get('project_display_name')
        if not project_display_name or project_display_name == '<project-display-name>':
            raise ValueError("project_display_name must be provided when creating a new project")
        project_id = create_project(
            data_science_client, 
            config['compartment_id'], 
            project_display_name, 
            config.get('project_description', '')
        )
    
    # Create provenance details
    provenance_config = config.get('provenance', {})
    provenance_details = CreateModelProvenanceDetails(
        repository_url=provenance_config.get('repository_url', ''),
        git_branch=provenance_config.get('git_branch', ''),
        git_commit=provenance_config.get('git_commit', ''),
        script_dir=provenance_config.get('script_dir', ''),
        training_script=provenance_config.get('training_script', ''),
        training_id=provenance_config.get('training_id', '')
    )
    
    # Create model details
    model_details_kwargs = {
        'compartment_id': config['compartment_id'],
        'project_id': project_id,
        'display_name': config['display_name'],
        'description': config.get('description', '')
    }
    
    # Only add optional parameters if they have values
    custom_metadata = config.get('custom_metadata_list', [])
    if custom_metadata:
        model_details_kwargs['custom_metadata_list'] = custom_metadata
    
    defined_metadata = config.get('defined_metadata_list', [])
    if defined_metadata:
        model_details_kwargs['defined_metadata_list'] = defined_metadata
    
    input_schema = config.get('input_schema_str', '')
    if input_schema:
        model_details_kwargs['input_schema'] = input_schema
    
    output_schema = config.get('output_schema_str', '')
    if output_schema:
        model_details_kwargs['output_schema'] = output_schema
    
    model_details = CreateModelDetails(**model_details_kwargs)
    
    # Debug: print model details
    print(f"Creating model: {config['display_name']}")
    print(f"Model details: {model_details_kwargs}")
    
    model = data_science_client.create_model(model_details)
    model_id = model.data.id
    print(f"Model created with ID: {model_id}")
    
    # Add provenance (if details provided)
    if any([provenance_details.repository_url, provenance_details.git_branch, 
            provenance_details.git_commit, provenance_details.training_script]):
        print("Adding model provenance...")
        data_science_client.create_model_provenance(model_id, provenance_details)
        print("Provenance added successfully")
    
    # Upload the artifact (zip file)
    zip_file_path = config['zip_file_path']
    print(f"Uploading artifact from: {zip_file_path}")
    
    # Check if file exists
    if not os.path.exists(zip_file_path):
        raise FileNotFoundError(f"Zip file not found: {zip_file_path}")
    
    # Get the filename from the path
    filename = os.path.basename(zip_file_path)
    
    # Read and upload the zip file
    with open(zip_file_path, 'rb') as artifact_file:
        artifact_bytes = artifact_file.read()
        data_science_client.create_model_artifact(
            model_id, 
            artifact_bytes, 
            content_disposition=f'attachment; filename="{filename}"'
        )
    
    print(f"Artifact uploaded successfully!")
    print(f"Model ID: {model_id}")
    # Create model deployment
    deployment_display_name = config.get('deployment_display_name')
    if deployment_display_name and deployment_display_name != '<deployment-display-name>':
        deployment_id = create_model_deployment(
            data_science_client,
            model_id,
            project_id,
            config['compartment_id'],
            deployment_display_name,
            config.get('deployment_description', ''),
            config
        )
        print(f"Deployment ID: {deployment_id}")
    
    return model_id


if __name__ == "__main__":
    try:
        model_id = upload_and_deploy_model()
        print("\nUpload and deployment setup completed successfully!")
    except Exception as e:
        print(f"\nError during upload: {e}")
        raise
