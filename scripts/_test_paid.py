import urllib.request, json

key = open('.env').read().split('=', 1)[1].strip()
modelo = 'qwen/qwen3-coder-30b-a3b-instruct'

payload = json.dumps({
    'model': modelo,
    'messages': [
        {
            'role': 'system',
            'content': 'Eres un experto en Python. Respondes ÚNICAMENTE con código Python limpio. Sin markdown. Solo código.'
        },
        {
            'role': 'user',
            'content': 'Escribe 2 funciones Python: una que implemente merge sort con type hints y docstring, y otra que verifique si una lista está ordenada.'
        }
    ],
    'max_tokens': 400,
    'temperature': 0.3,
}).encode()

req = urllib.request.Request(
    'https://openrouter.ai/api/v1/chat/completions',
    data=payload,
    headers={
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://github.com/lucasmella-stack/PAMPAr-Coder',
        'X-Title': 'PAMPAr-Coder Distillation',
    },
    method='POST'
)

try:
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
        if 'choices' in data:
            usage = data.get('usage', {})
            texto = data['choices'][0]['message']['content']
            print(f"OK — {usage.get('prompt_tokens',0)} in / {usage.get('completion_tokens',0)} out tokens")
            print('-' * 60)
            print(texto[:800])
        else:
            print('ERROR body:', str(data)[:300])
except Exception as e:
    print('EXC:', e)
