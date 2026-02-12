import subprocess, json

API = "REDACTED"
URL = f"https://api.runpod.io/graphql?api_key={API}"

# Terminate stuck pod
q = {"query": 'mutation { podTerminate(input: { podId: "19lqkdmf4w0kyv" }) }'}
r = subprocess.run(["curl", "-s", "-H", "Content-Type: application/json", "-d", json.dumps(q), URL], capture_output=True, text=True)
print("Terminate:", r.stdout)

# Create new with lightweight image
q2 = {"query": 'mutation { podFindAndDeployOnDemand(input: { name: "pampar-train", gpuTypeId: "NVIDIA RTX A5000", gpuCount: 1, cloudType: SECURE, containerDiskInGb: 40, volumeInGb: 50, imageName: "runpod/base:0.6.2-cuda12.2.0", ports: "22/tcp" }) { id desiredStatus machine { podHostId gpuDisplayName } } }'}
r2 = subprocess.run(["curl", "-s", "-H", "Content-Type: application/json", "-d", json.dumps(q2), URL], capture_output=True, text=True)
print("Create:", r2.stdout)
