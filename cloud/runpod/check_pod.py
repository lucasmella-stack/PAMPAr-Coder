#!/usr/bin/env python3
"""Check RunPod pod status."""
import os
import urllib.request
import json
import ssl

API_KEY = os.environ.get("RUNPOD_API_KEY", "")
QUERY = """{ myself { pods { id name desiredStatus runtime { uptimeInSeconds gpus { gpuUtilPerc } ports { ip publicPort privatePort } } } } }"""

url = f"https://api.runpod.io/graphql?api_key={API_KEY}"
body = json.dumps({"query": QUERY}).encode()
ctx = ssl.create_default_context()
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
try:
    resp = json.loads(urllib.request.urlopen(req, timeout=10, context=ctx).read())
except Exception as e:
    print(f"Error with default SSL: {e}")
    # Try without SSL verification
    ctx = ssl._create_unverified_context()
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=10, context=ctx).read())
    except Exception as e2:
        print(f"Error without SSL: {e2}")
        # Try POST via different method
        import http.client
        conn = http.client.HTTPSConnection("api.runpod.io", timeout=10, context=ssl._create_unverified_context())
        conn.request("POST", f"/graphql?api_key={API_KEY}", body=body, headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        print(f"Direct HTTPS status: {r.status} {r.reason}")
        raw = r.read().decode()
        print(f"Body: {raw[:500]}")
        if r.status == 200:
            resp = json.loads(raw)
        else:
            exit(1)

pods = resp["data"]["myself"]["pods"]
if not pods:
    print("No pods found!")
else:
    for p in pods:
        rt = p.get("runtime") or {}
        ports = rt.get("ports") or []
        ssh = [x for x in ports if x.get("privatePort") == 22]
        gpus = rt.get("gpus") or []
        
        print(f"Pod: {p['id']} ({p['name']})")
        print(f"  Status: {p['desiredStatus']}")
        print(f"  Uptime: {rt.get('uptimeInSeconds', 0)}s")
        print(f"  GPU util: {[g.get('gpuUtilPerc', '?') for g in gpus]}")
        if ssh:
            print(f"  SSH: {ssh[0]['ip']}:{ssh[0]['publicPort']}")
        else:
            print("  SSH: N/A (pod may be stopped)")
