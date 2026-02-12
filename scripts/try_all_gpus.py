import subprocess, json, time, sys

API = "REDACTED"
URL = f"https://api.runpod.io/graphql?api_key={API}"

# 1. Terminate current stuck pod
print("Terminating stuck pod...")
q = {"query": 'mutation { podTerminate(input: { podId: "coexq8gdno2lxh" }) }'}
r = subprocess.run(["curl", "-s", "-H", "Content-Type: application/json", "-d", json.dumps(q), URL], capture_output=True, text=True)
print(f"  {r.stdout.strip()}")

# 2. Try multiple GPUs with quick 2-min timeout each
GPUS = [
    ("NVIDIA GeForce RTX 4090", "SECURE"),
    ("NVIDIA A40", "SECURE"),
    ("NVIDIA RTX A6000", "SECURE"),
    ("NVIDIA L4", "SECURE"),
    ("NVIDIA GeForce RTX 3090 Ti", "SECURE"),
    ("Tesla V100-SXM2-32GB", "SECURE"),
]

for gpu_id, cloud in GPUS:
    print(f"\nTrying {gpu_id} ({cloud})...")
    q = {"query": f'mutation {{ podFindAndDeployOnDemand(input: {{ name: "pampar-train", gpuTypeId: "{gpu_id}", gpuCount: 1, cloudType: {cloud}, containerDiskInGb: 40, volumeInGb: 50, imageName: "runpod/base:0.6.2-cuda12.2.0", ports: "22/tcp" }}) {{ id desiredStatus machine {{ podHostId gpuDisplayName }} }} }}'}
    r = subprocess.run(["curl", "-s", "-H", "Content-Type: application/json", "-d", json.dumps(q), URL], capture_output=True, text=True)
    resp = json.loads(r.stdout)
    
    if "errors" in resp:
        print(f"  No availability: {resp['errors'][0]['message'][:60]}")
        continue
    
    pod = resp["data"]["podFindAndDeployOnDemand"]
    pod_id = pod["id"]
    gpu = pod["machine"]["gpuDisplayName"]
    print(f"  Pod {pod_id} created on {gpu}. Waiting for startup...")
    
    # Wait up to 2 minutes
    for i in range(12):
        time.sleep(10)
        q2 = {"query": f'{{ pod(input: {{ podId: "{pod_id}" }}) {{ id desiredStatus runtime {{ uptimeInSeconds ports {{ ip isIpPublic publicPort privatePort }} }} }} }}'}
        r2 = subprocess.run(["curl", "-s", "-H", "Content-Type: application/json", "-d", json.dumps(q2), URL], capture_output=True, text=True)
        data = json.loads(r2.stdout)
        runtime = data["data"]["pod"].get("runtime")
        if runtime:
            ports = runtime.get("ports", [])
            ssh_port = None
            for p in ports:
                if p["privatePort"] == 22:
                    ssh_port = p
            if ssh_port:
                print(f"  SUCCESS! SSH: root@{ssh_port['ip']} -p {ssh_port['publicPort']}")
                print(f"  Pod ID: {pod_id}")
                sys.exit(0)
            else:
                print(f"  Runtime found but no SSH port. Ports: {ports}")
                sys.exit(0)
        print(f"    [{i+1}/12] Still starting...", flush=True)
    
    # Timed out, terminate and try next
    print(f"  Timed out. Terminating {pod_id}...")
    qt = {"query": f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}'}
    subprocess.run(["curl", "-s", "-H", "Content-Type: application/json", "-d", json.dumps(qt), URL], capture_output=True, text=True)

print("\nAll GPU types exhausted. RunPod may have platform issues right now.")
print("Check balance:")
qb = {"query": "{ myself { currentSpendPerHr clientBalance } }"}
rb = subprocess.run(["curl", "-s", "-H", "Content-Type: application/json", "-d", json.dumps(qb), URL], capture_output=True, text=True)
print(f"  {rb.stdout.strip()}")
