import subprocess, json, sys

API_KEY = "REDACTED"
URL = f"https://api.runpod.io/graphql?api_key={API_KEY}"

# Try these GPUs in order of preference (price/performance)
GPU_ATTEMPTS = [
    ("NVIDIA RTX A5000", "SECURE"),
    ("NVIDIA GeForce RTX 3090 Ti", "SECURE"),
    ("NVIDIA GeForce RTX 3090 Ti", "COMMUNITY"),
    ("NVIDIA A30", "SECURE"),
    ("NVIDIA A30", "COMMUNITY"),
    ("NVIDIA RTX A5000", "COMMUNITY"),
    ("NVIDIA RTX A6000", "COMMUNITY"),
    ("NVIDIA RTX A6000", "SECURE"),
    ("NVIDIA GeForce RTX 4090", "COMMUNITY"),
    ("NVIDIA GeForce RTX 4090", "SECURE"),
    ("NVIDIA A40", "COMMUNITY"),
    ("NVIDIA A40", "SECURE"),
    ("NVIDIA L4", "SECURE"),
    ("NVIDIA GeForce RTX 3090", "SECURE"),
    ("NVIDIA GeForce RTX 3090", "COMMUNITY"),
    ("NVIDIA RTX PRO 4500 Blackwell", "COMMUNITY"),
    ("NVIDIA RTX PRO 4500 Blackwell", "SECURE"),
    ("Tesla V100-SXM2-32GB", "COMMUNITY"),
    ("Tesla V100-SXM2-32GB", "SECURE"),
]

for gpu_id, cloud in GPU_ATTEMPTS:
    query = {
        "query": f'mutation {{ podFindAndDeployOnDemand(input: {{ name: "pampar-train", gpuTypeId: "{gpu_id}", gpuCount: 1, cloudType: {cloud}, containerDiskInGb: 40, volumeInGb: 50, imageName: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04", ports: "22/tcp,8888/http" }}) {{ id desiredStatus machine {{ podHostId gpuDisplayName }} }} }}'
    }
    print(f"Trying {gpu_id} ({cloud})...", end=" ", flush=True)
    result = subprocess.run(
        ["curl", "-s", "-H", "Content-Type: application/json", "-d", json.dumps(query), URL],
        capture_output=True, text=True
    )
    resp = json.loads(result.stdout)
    if "errors" in resp:
        print(f"FAIL: {resp['errors'][0]['message'][:60]}")
    else:
        pod = resp["data"]["podFindAndDeployOnDemand"]
        print(f"SUCCESS! Pod: {pod['id']} GPU: {pod['machine']['gpuDisplayName']}")
        sys.exit(0)

print("\nNo GPUs available anywhere!")
sys.exit(1)
