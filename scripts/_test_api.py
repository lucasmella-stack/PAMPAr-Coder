import urllib.request, urllib.error, json, time

key = open('.env').read().split('=', 1)[1].strip()

modelos = [
    'qwen/qwen3-coder:free',
    'meta-llama/llama-3.3-70b-instruct:free',
    'mistralai/mistral-small-3.1-24b-instruct:free',
]

for modelo in modelos:
    nombre = modelo.split('/')[1]
    payload = json.dumps({
        'model': modelo,
        'messages': [{'role': 'user', 'content': 'Write a Python function that returns the sum of two numbers. Only code.'}],
        'max_tokens': 60
    }).encode()
    req = urllib.request.Request(
        'https://openrouter.ai/api/v1/chat/completions',
        data=payload,
        headers={
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://github.com/lucasmella-stack/PAMPAr-Coder',
        },
        method='POST'
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
            if 'choices' in data:
                print(f'OK [{nombre}]:', data['choices'][0]['message']['content'][:80])
            else:
                print(f'ERR body [{nombre}]:', str(data)[:250])
    except urllib.error.HTTPError as e:
        print(f'HTTP {e.code} [{nombre}]:', e.read().decode()[:200])
    except Exception as e:
        print(f'EXC [{nombre}]:', e)
    time.sleep(5)
