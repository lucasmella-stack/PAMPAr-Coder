import subprocess, json, time, sys

API = "REDACTED"
URL = f"https://api.runpod.io/graphql?api_key={API}"
POD_ID = "coexq8gdno2lxh"

print(f"Waiting for pod {POD_ID} to start...")
for i in range(30):  # up to 5 minutes
    q = {"query": f'{{ pod(input: {{ podId: "{POD_ID}" }}) {{ id desiredStatus runtime {{ uptimeInSeconds ports {{ ip isIpPublic publicPort privatePort }} }} }} }}'}
    r = subprocess.run(["curl", "-s", "-H", "Content-Type: application/json", "-d", json.dumps(q), URL], capture_output=True, text=True)
    data = json.loads(r.stdout)
    pod = data["data"]["pod"]
    runtime = pod.get("runtime")
    if runtime is not None:
        print(f"\nPod is UP! Uptime: {runtime.get('uptimeInSeconds', 0)}s")
        ports = runtime.get("ports", [])
        for p in ports:
            print(f"  Port {p['privatePort']} -> {p['ip']}:{p['publicPort']} (public={p['isIpPublic']})")
            if p['privatePort'] == 22:
                print(f"\n  SSH: ssh -o StrictHostKeyChecking=no root@{p['ip']} -p {p['publicPort']}")
        sys.exit(0)
    else:
        print(f"  [{i+1}/30] Still starting... (status={pod['desiredStatus']})", flush=True)
        time.sleep(10)

print("\nPod failed to start in 5 minutes!")
sys.exit(1)
