import json, sys
data = json.load(sys.stdin)
gpus = [g for g in data['data']['gpuTypes'] if g['memoryInGb'] >= 24 and (g.get('secureCloud') or g.get('communityCloud'))]
gpus.sort(key=lambda g: g.get('communityPrice') or g.get('securePrice') or 999)
for g in gpus[:15]:
    print(f"{g['id']:30} {g['displayName']:25} {g['memoryInGb']}GB  sec={g.get('securePrice')} com={g.get('communityPrice')}")
