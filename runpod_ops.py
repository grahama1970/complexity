from docker import from_env
from docker.errors import DockerException

def build_and_push_image():
    """Build and push Docker image using compose"""
    try:
        # Authenticate
        client = from_env()
        client.login(
            username=os.getenv("DOCKERHUB_USERNAME"),
            password=os.getenv("DOCKERHUB_TOKEN")
        )

        # Build and push
        client.images.build(
            path=".",
            tag=f"{os.getenv('DOCKERHUB_USERNAME')}/complexity-training:latest",
            dockerfile="Dockerfile"
        )
        client.images.push(
            repository=f"{os.getenv('DOCKERHUB_USERNAME')}/complexity-training",
            tag="latest"
        )
        
    except DockerException as e:
        logger.error(f"Docker operation failed: {e}")
        raise

# Modify your pod config to use the built image
def get_pod_config():
    return {
        "image_name": f"{os.getenv('DOCKERHUB_USERNAME')}/complexity-training:latest",
        "gpu_type_id": "NVIDIA RTX 3090",
        "cloud_type": "SECURE",
        "env": {
            "HF_TOKEN": os.getenv("HF_TOKEN"),
            "PYTHONPATH": "/app/server/src"
        },
        "volume_in_gb": 50,
        "container_disk_in_gb": 20,
        "ports": "8000/http"
    } 