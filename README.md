# OCI TEI Deployment Package

This package deploys a Hugging Face Text Embeddings Inference (TEI) container to OCI Data Science.

| File | Purpose |
| --- | --- |
| `oci_tei_deployment.py` | Registers a model artifact, uploads it, then creates its deployment. |
| `tei_deploy.py` | Creates a deployment for an already registered model OCID. |
| `config.yaml` | Deployment configuration: OCI compartment/project, artifact, OCIR image, shape, and TEI arguments. |

## Customer setup

1. Install Python 3.9 or later. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the OCI Python SDK and YAML dependency. Custom HTTP endpoints such as `/embed` require OCI SDK 2.182.0 or later:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install --upgrade 'oci>=2.182.0' pyyaml
   python -c "import oci; print(oci.__version__)"
   ```

3. Configure OCI CLI/API-key authentication in `~/.oci/config`.
4. Update `config.yaml` for the customer tenancy. In particular replace `compartment_id`, `project_id` (or project display settings), `zip_file_path`, and `container_image`.
5. Ensure the configured OCIR image is accessible to the OCI Data Science deployment and the artifact zip contains the model at the TEI `--model-id` path.

## Verify the model artifact layout

Use the manually uploaded ZIP that contains the complete model. Its top-level directory must be `bge-small-en-v1.5`.

```text
bge-small-en-v1.5/
├── config.json
├── model.safetensors
├── tokenizer.json
├── sentence_bert_config.json
├── config_sentence_transformers.json
├── modules.json
└── 1_Pooling/
    └── config.json
```

Validate the ZIP before using it:

```bash
unzip -l bge-small-en-v1.5.zip
```

The archive must not contain a second ZIP file in place of the model files. Keep the top-level `bge-small-en-v1.5/` directory because the TEI startup command uses that exact path.

## Push the wrapper image and record its immutable digest

The configured container name is `phx.ocir.io/namespace/tembed:1.9`. Log in to the Phoenix OCIR registry and push the new tag:

```bash
sudo podman login phx.ocir.io

sudo podman push \
  --digestfile /tmp/tembed-1.9.1.digest \
  phx.ocir.io/namespace/tembed:1.9.1

cat /tmp/tembed-1.9.1.digest
```

Save the returned `sha256:...` digest. The new deployment must use the immutable image reference, not only the mutable tag:

```yaml
container_image: 'phx.ocir.io/namespace/tembed:1.9.1@sha256:<image-digest>'
```

## Run

Upload/register the artifact and create a deployment:

```bash
python3 oci_tei_deployment.py
```

Create a deployment for a model already in OCI Model Catalog:

```bash
python3 tei_deploy.py --model-id <model-ocid> --wait
```

The configured TEI endpoint is invoked with `POST <model-deployment-url>/predict/embed`. OCI removes the `/predict` prefix and forwards the request to the container's `POST /embed`. Both the server and health-check ports are `8080`.

## Before committing

Do not commit OCI API keys, OCI config files, private image credentials, or customer-specific values you do not intend to disclose. `config.yaml` contains deployment identifiers and local paths; replace them with customer-safe values before publishing.
