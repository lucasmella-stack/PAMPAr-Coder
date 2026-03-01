import urllib.request, json, time

key = open('.env').read().split('=', 1)[1].strip()

# Buscar modelos de código con buen precio
req = urllib.request.Request(
    'https://openrouter.ai/api/v1/models',
    headers={'Authorization': f'Bearer {key}'},
    method='GET'
)
with urllib.request.urlopen(req, timeout=15) as r:
    data = json.loads(r.read())

# Filtrar modelos relevantes para código
keywords = ['coder', 'code', 'deepseek', 'qwen', 'starcoder', 'codestral']
candidatos = []
for m in data['data']:
    mid = m['id'].lower()
    if any(k in mid for k in keywords) and ':free' not in m['id']:
        price_in = float(m.get('pricing', {}).get('prompt', '999') or 999)
        price_out = float(m.get('pricing', {}).get('completion', '999') or 999)
        ctx = m.get('context_length', 0)
        candidatos.append((price_in + price_out, m['id'], price_in, price_out, ctx))

candidatos.sort()
print(f"{'MODELO':<55} {'$/M in':<10} {'$/M out':<10} {'ctx'}")
print('-' * 90)
for total, mid, pin, pout, ctx in candidatos[:20]:
    # Calcular costo por 1000 ejemplos (400 in + 600 out tokens cada uno)
    costo_1k = (400 * pin + 600 * pout) / 1000
    print(f"{mid:<55} {pin*1e6:<10.3f} {pout*1e6:<10.3f} ctx={ctx//1000}k  ~${costo_1k:.4f}/1k_ejs")
