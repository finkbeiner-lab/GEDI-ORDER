import wandb
import os
import argparse

def download_wandb_models(entity, project, run_id, download_path, artifact_name=None):
    """
    Download model artifacts from W&B run to specified path.

    Args:
        entity: W&B entity/username
        project: W&B project name
        run_id: Specific run ID to download from
        download_path: Local directory to save models
        artifact_name: Specific artifact name (optional)
    """
    api = wandb.Api()

    # Create download directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)

    # Get the specific run
    run = api.run(f"{entity}/{project}/{run_id}")
    print(f"Downloading from run: {run.name} (ID: {run.id})")
    print(f"Run URL: {run.url}")

    # Download artifacts logged to this run (models saved via wandb.log_artifact)
    artifacts = run.logged_artifacts()
    artifact_found = False
    for artifact in artifacts:
        if artifact_name and artifact.name.split(':')[0] != artifact_name:
            continue

        # Only download model artifacts
        if artifact.type in ['model', 'keras-model', 'pytorch-model', 'tensorflow-model']:
            print(f"Downloading model artifact: {artifact.name} (type: {artifact.type})")
            artifact.download(root=download_path)
            print(f"  Saved to: {download_path}")
            artifact_found = True
        else:
            print(f"Skipping non-model artifact: {artifact.name} (type: {artifact.type})")

    # Always check for model files attached directly to the run
    print("\nChecking for model files in run...")
    for file in run.files():
        if file.name.endswith(('.h5', '.keras', '.ckpt', '.pb', '.pth', '.pt', '.safetensors')):
            print(f"Downloading model file: {file.name}")
            file.download(root=download_path, replace=True)
            print(f"  Saved to: {os.path.join(download_path, file.name)}")
            artifact_found = True
        else:
            print(f"Skipping non-model file: {file.name}")

    if not artifact_found:
        print("Warning: no model files or artifacts found for this run.")

    print(f"\nAll models downloaded to: {download_path}")

def list_runs(entity, project, limit=10):
    """List recent runs from a W&B project."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", per_page=limit)

    print(f"\nRecent runs in {entity}/{project}:")
    print("-" * 60)
    for run in runs:
        print(f"ID: {run.id}")
        print(f"Name: {run.name}")
        print(f"State: {run.state}")
        print(f"Created: {run.created_at}")
        print("-" * 60)

    return runs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from W&B")
    parser.add_argument("--entity", type=str, default="swang202",
                        help="W&B entity/username")
    parser.add_argument("--project", type=str, default="TauKO_vs_KO+OE",
                        help="W&B project name")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Specific run ID to download from")
    parser.add_argument("--download_path", type=str,
                        default="/home/shijiewang/Documents/GEDI-ORDER/downloaded_models",
                        help="Path to save downloaded models")
    parser.add_argument("--artifact_name", type=str, default=None,
                        help="Specific artifact name to download")
    parser.add_argument("--list_runs", action="store_true",
                        help="List available runs without downloading")

    args = parser.parse_args()

    if args.list_runs:
        # Just list runs
        list_runs(args.entity, args.project)
    elif args.run_id:
        # Download from specific run
        download_wandb_models(
            entity=args.entity,
            project=args.project,
            run_id=args.run_id,
            download_path=args.download_path,
            artifact_name=args.artifact_name
        )
    else:
        # List runs and prompt for selection
        runs = list_runs(args.entity, args.project)
        if runs:
            print("\nTo download models from a specific run, use:")
            print(f"python {__file__} --run_id <RUN_ID> --download_path {args.download_path}")