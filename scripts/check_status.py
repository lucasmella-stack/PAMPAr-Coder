import subprocess, json
API = "REDACTED"
q = {"query": '{ pod(input: { podId: "19lqkdmf4w0kyv" }) { id desiredStatus runtime { uptimeInSeconds ports { ip isIpPublic publicPort privatePort } } } }'}
r = subprocess.run(["curl", "-s", "-H", "Content-Type: application/json", "-d", json.dumps(q), f"https://api.runpod.io/graphql?api_key={API}"], capture_output=True, text=True)
data = json.loads(r.stdout)
print(json.dumps(data, indent=2))
