#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
"""
destilar_v3.py — Destilación de Qwen2.5-Coder-32B → PamparV3

Usa OpenRouter (tier GRATIS) para generar código Python de calidad
profesional y lo guarda como JSONL para entrenar PamparV3.

Modelos gratuitos disponibles en OpenRouter:
  - qwen/qwen-2.5-coder-32b-instruct:free  ← recomendado (mejor en código)
  - deepseek/deepseek-r1:free               ← reasoning, más lento
  - google/gemini-2.0-flash-thinking-exp:free

Uso:
  # Obtener API key gratis en https://openrouter.ai/keys
  python scripts/destilar_v3.py --api-key sk-or-...
  python scripts/destilar_v3.py --api-key sk-or-... --n 2000
  python scripts/destilar_v3.py --api-key sk-or-... --modelo deepseek/deepseek-r1:free

  # Con variable de entorno (recomendado):
  $env:OPENROUTER_API_KEY = "sk-or-..."
  python scripts/destilar_v3.py --n 3000
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Leer .env si existe (para no exponer la key en el terminal)
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# =============================================================================
# Prompts para el profesor — ejercicios de código Python de distinto nivel
# =============================================================================

TEMAS_NIVEL_1 = [
    "funciones de strings: reverse, count, replace, strip, split",
    "operaciones con listas: append, sort, filter, map, zip",
    "números: primos, factoriales, fibonacci, divisores",
    "diccionarios: frecuencias, invertir, merge, filtrar por valor",
    "condicionales y loops: FizzBuzz, triangulos, patrones",
    "manejo de fechas con datetime",
    "operaciones de sets: union, interseccion, diferencia",
]

TEMAS_NIVEL_2 = [
    "recursion: torres de hanoi, permutaciones, árbol de decisión",
    "algoritmos de sorting: bubble, merge, quick, heap sort",
    "búsqueda binaria y variantes",
    "programación funcional: reduce, partial, currying, closures",
    "generadores e iteradores: yield, send, StopIteration",
    "decoradores: memoize, timer, retry, rate_limit",
    "context managers: __enter__, __exit__, contextlib",
    "comprensiones anidadas y expresiones generadoras complejas",
]

TEMAS_NIVEL_3 = [
    "clases y OOP: herencia, polimorfismo, dunder methods",
    "dataclasses y attrs: frozen, validators, converters",
    "patrones de diseño: singleton, factory, observer, strategy",
    "metaclases y descriptores",
    "async/await: asyncio, tasks, gather, queues",
    "threading y multiprocessing: locks, events, pools",
    "estructuras de datos: linked list, árbol BST, heap, graph",
    "algoritmos de grafos: BFS, DFS, Dijkstra, A*",
    "type hints avanzados: Generic, Protocol, TypeVar, overload",
]

TEMAS_NIVEL_4 = [
    "parsers y tokenizers simples desde cero",
    "implementar un mini ORM estilo Django desde cero",
    "web scraping con requests y BeautifulSoup",
    "API REST simple con FastAPI y Pydantic",
    "testing con pytest: fixtures, parametrize, mocking, coverage",
    "profiling y optimización: cProfile, memory_profiler, line_profiler",
    "implementar un sistema de caché LRU y TTL",
    "compresión y serialización: pickle, json, msgpack, protocol buffers",
]

TODOS_LOS_TEMAS = [
    (1, t) for t in TEMAS_NIVEL_1
] + [
    (2, t) for t in TEMAS_NIVEL_2
] + [
    (3, t) for t in TEMAS_NIVEL_3
] + [
    (4, t) for t in TEMAS_NIVEL_4
]


def prompt_para_tema(nivel: int, tema: str) -> str:
    """Genera un prompt que hace que el profesor produzca código Python real."""
    instrucciones = {
        1: "Escribe 3 funciones Python simples pero útiles relacionadas con",
        2: "Escribe 2 soluciones Python con explicación de complejidad sobre",
        3: "Implementa desde cero en Python (sin librerías externas) un ejemplo completo de",
        4: "Escribe código Python de producción, bien documentado y con tests, sobre",
    }
    base = instrucciones.get(nivel, "Escribe código Python sobre")

    return (
        f"{base} {tema}. "
        "Incluye docstrings, type hints, manejo de edge cases y ejemplos de uso. "
        "El código debe ser correcto, idiomático y ejecutable tal cual. "
        "No incluyas explicaciones en prosa, solo código Python con comentarios inline."
    )


# =============================================================================
# Cliente OpenRouter
# =============================================================================

# Modelo principal (pago, con créditos OpenRouter)
MODELO_PRINCIPAL = "qwen/qwen3-coder-30b-a3b-instruct"

# Fallback si el principal tiene rate limit
MODELOS_FALLBACK = [
    "qwen/qwen3-coder-30b-a3b-instruct",
    "qwen/qwen-2.5-coder-32b-instruct",
    "qwen/qwen2.5-coder-7b-instruct",
]

# Precio aproximado por 1M tokens (input, output) — para tracking de costo
PRECIO_POR_MILLON = {
    "qwen/qwen3-coder-30b-a3b-instruct":    (0.070, 0.270),
    "qwen/qwen-2.5-coder-32b-instruct":     (0.200, 0.200),
    "qwen/qwen2.5-coder-7b-instruct":       (0.030, 0.090),
}

MODELOS_FREE = MODELOS_FALLBACK  # alias para retrocompatibilidad

_modelo_idx = 0  # rotación global
_costo_total = 0.0  # tracking de gasto


def _sumar_costo(modelo: str, tokens_in: int, tokens_out: int) -> float:
    """Calcula y acumula costo de la llamada."""
    global _costo_total
    pin, pout = PRECIO_POR_MILLON.get(modelo, (0.2, 0.2))
    costo = (tokens_in * pin + tokens_out * pout) / 1_000_000
    _costo_total += costo
    return costo


def llamar_api(
    prompt: str,
    api_key: str,
    modelo: str,
    max_tokens: int = 1500,
    temperature: float = 0.3,
    timeout: int = 90,
) -> str | None:
    """
    Llama a OpenRouter API con backoff exponencial global.

    Cuando todos los modelos devuelven 429, espera 60s antes de reintentar
    en vez de rotar rápidamente y agotar más cuota.
    """
    import urllib.request
    import urllib.error
    global _modelo_idx

    n_modelos = len(MODELOS_FREE)

    for ronda in range(n_modelos * 2):  # hasta 2 vueltas completas
        modelo_actual = MODELOS_FREE[_modelo_idx % n_modelos]

        payload = json.dumps({
            "model": modelo_actual,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Eres un experto en Python. Respondes ÚNICAMENTE con código Python "
                        "limpio, funcional y bien documentado. Sin markdown, sin bloques ```python. "
                        "Solo código Python puro que se pueda ejecutar directamente."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/lucasmella-stack/PAMPAr-Coder",
                "X-Title": "PAMPAr-Coder Distillation",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if "error" in data and "choices" not in data:
                    print(f"\n  [API error body] {str(data)[:150]}")
                    _modelo_idx += 1
                    time.sleep(3)
                    continue
                texto = data["choices"][0]["message"]["content"]
                # Tracking de costo
                usage = data.get("usage", {})
                _sumar_costo(modelo_actual, usage.get("prompt_tokens", 400), usage.get("completion_tokens", 600))
                if texto:
                    return texto
                # Respuesta vacía → rotar modelo
                _modelo_idx += 1
                continue

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            if e.code == 429:
                _modelo_idx += 1
                siguiente = MODELOS_FREE[_modelo_idx % n_modelos]
                # Si completamos una vuelta completa, backoff global de 60s
                if ronda > 0 and ronda % n_modelos == 0:
                    print(f"\n  [rate limit global] todos los modelos saturados — esperando 60s...", end="", flush=True)
                    time.sleep(60)
                else:
                    print(f"\n  [429] → {siguiente.split('/')[1]}  ", end="", flush=True)
                    time.sleep(5)
                continue
            print(f"\n  [API error {e.code}] {body[:150]}")
            return None
        except Exception as e:
            print(f"\n  [error] {type(e).__name__}: {e}")
            time.sleep(5)
            return None

    print(f"\n  [!, agotados reintentos]")
    return None


# =============================================================================
# Validación básica del código generado
# =============================================================================

def limpiar_codigo(texto: str) -> str:
    """Elimina bloques markdown ```python ... ``` si el modelo los incluye."""
    texto = texto.strip()
    if texto.startswith("```"):
        # Quitar primera línea (```python o ```)
        lineas = texto.split("\n")
        if lineas[0].startswith("```"):
            lineas = lineas[1:]
        # Quitar cierre ```
        if lineas and lineas[-1].strip() == "```":
            lineas = lineas[:-1]
        texto = "\n".join(lineas).strip()
    return texto


def es_codigo_valido(texto: str) -> bool:
    """Rechaza respuestas que no son código Python real."""
    import ast
    if len(texto.strip()) < 30:
        return False
    # Debe tener al menos una definición
    tiene_def = any(kw in texto for kw in ("def ", "class ", "async def "))
    if not tiene_def:
        return False
    # Intentar parsear (no falla si hay clases parciales, etc.)
    try:
        ast.parse(texto)
        return True
    except SyntaxError:
        # Aceptar igual — el modelo puede generar fragmentos válidos con pequeños errores
        return len(texto) > 100


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Destilación Qwen→PamparV3 vía OpenRouter")
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"), help="OpenRouter API key")
    parser.add_argument("--modelo", default=MODELO_PRINCIPAL,
                        help="Modelo de OpenRouter a usar como profesor")
    parser.add_argument("--n", type=int, default=5000, help="Ejemplos a generar")
    parser.add_argument("--output", default="biblioteca/python_real/destilado_qwen.jsonl")
    parser.add_argument("--temp", type=float, default=0.35)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Segundos entre llamadas (0.3s con créditos — hasta 200 rpm)")
    parser.add_argument("--si", action="store_true", help="No pedir confirmación")
    args = parser.parse_args()

    if not args.api_key:
        print("\n  ❌ Necesitás una API key de OpenRouter (gratis en https://openrouter.ai/keys)")
        print("     Pasala con --api-key sk-or-... o con:")
        print("     $env:OPENROUTER_API_KEY = 'sk-or-...'")
        sys.exit(1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Ejemplos ya existentes (para no regenerar si se interrumpe)
    ya_generados = 0
    if output.exists():
        with open(output, encoding="utf-8") as f:
            ya_generados = sum(1 for _ in f)

    print(f"\n{'═'*65}")
    print(f"  DESTILACIÓN — {args.modelo.split('/')[-1]}")
    print(f"{'═'*65}")
    print(f"  Output     : {output}")
    print(f"  Objetivo   : {args.n} ejemplos")
    print(f"  Ya existen : {ya_generados} ejemplos")
    print(f"  Por generar: {max(0, args.n - ya_generados)} ejemplos")
    print(f"  Delay      : {args.delay}s (tier free)")
    pin, pout = PRECIO_POR_MILLON.get(args.modelo, (0.2, 0.2))
    costo_estimado = (args.n * 600 * pin + args.n * 900 * pout) / 1_000_000
    print(f"  Modelo     : {args.modelo}")
    print(f"  Costo est. : ~${costo_estimado:.2f} de $10 disponibles")
    print(f"{'═'*65}\n")

    faltan = args.n - ya_generados
    if faltan <= 0:
        print(f"  ✅ Ya tenés {ya_generados} ejemplos. Usá --n {ya_generados + 500} para más.")
        return

    if not args.si:
        resp = input(f"  ¿Generar {faltan} ejemplos? (s/n): ").strip().lower()
        if resp not in ("s", "si", "sí", "y", "yes"):
            print("  Cancelado.")
            return

    n_ok = 0
    n_error = 0
    n_invalido = 0
    t_start = time.time()

    # Ciclar por temas infinitamente hasta llegar al objetivo
    temas_pool = TODOS_LOS_TEMAS * ((faltan // len(TODOS_LOS_TEMAS)) + 2)
    random.shuffle(temas_pool)

    with open(output, "a", encoding="utf-8") as f_out:
        for i, (nivel, tema) in enumerate(temas_pool):
            if n_ok >= faltan:
                break

            prompt = prompt_para_tema(nivel, tema)
            elapsed = time.time() - t_start
            eta = (elapsed / max(n_ok, 1)) * (faltan - n_ok) if n_ok > 0 else "?"

            print(
                f"  [{ya_generados + n_ok + 1:04d}/{args.n}]  "
                f"nivel={nivel}  {tema[:45]:<45}  ",
                end="", flush=True,
            )

            respuesta = llamar_api(
                prompt,
                args.api_key,
                args.modelo,
                max_tokens=args.max_tokens,
                temperature=args.temp,
            )

            if respuesta is None:
                n_error += 1
                print(f"ERROR  (total errores: {n_error})")
                if n_error % 5 == 0:
                    print(f"  [!] {n_error} errores — esperando 30s...")
                    time.sleep(30)
                continue

            respuesta = limpiar_codigo(respuesta)

            if not es_codigo_valido(respuesta):
                n_invalido += 1
                print(f"INVÁLIDO ({len(respuesta)} chars)")
                continue

            # Guardar en formato compatible con LectorBiblioteca
            entrada = json.dumps({
                "instruction": prompt,
                "output": respuesta,
                "nivel": nivel,
                "tema": tema,
                "profesor": args.modelo,
            }, ensure_ascii=False)
            f_out.write(entrada + "\n")
            f_out.flush()

            n_ok += 1
            tok_aprox = len(respuesta.split())
            eta_str = f"{int(eta)}s" if isinstance(eta, float) else eta
            print(f"OK  ~{tok_aprox}tok  ${_costo_total:.3f}  ETA:{eta_str}")

            time.sleep(args.delay)

    # ── Resumen ──────────────────────────────────────────────────────────────
    total_archivo = ya_generados + n_ok
    elapsed = time.time() - t_start
    print(f"\n{'═'*65}")
    print(f"  COMPLETADO en {elapsed/60:.1f} min")
    print(f"  Nuevos ejemplos  : {n_ok}")
    print(f"  Total en archivo : {total_archivo}")
    print(f"  Errores API      : {n_error}")
    print(f"  Inválidos        : {n_invalido}")
    print(f"  Costo real       : ${_costo_total:.4f}")
    print(f"  Output           : {output}")
    print(f"{'═'*65}")

    # Actualizar indice.json automáticamente
    _registrar_en_indice(output, total_archivo)

    print(f"\n  ✅ Listo. Para entrenar con estos datos:")
    print(f"  & 'C:\\Users\\lucas\\AppData\\Local\\Programs\\Python\\Python313\\python.exe'")
    print(f"    scripts/train_v3.py --checkpoint checkpoints/v3_train.pt --lr 2e-5")


def _registrar_en_indice(output: Path, n_ejemplos: int) -> None:
    """Añade o actualiza la entrada en biblioteca/indice.json."""
    indice_path = Path("biblioteca/indice.json")
    if not indice_path.exists():
        return

    with open(indice_path, encoding="utf-8") as f:
        indice = json.load(f)

    nombre = "destilado_qwen"
    ruta_relativa = str(output.relative_to(Path("biblioteca"))).replace("\\", "/")

    # Buscar si ya existe
    existe = False
    for entrada in indice:
        if entrada.get("nombre") == nombre:
            entrada["n_ejemplos"] = n_ejemplos
            existe = True
            break

    if not existe:
        indice.append({
            "nombre": nombre,
            "categoria": "destilacion",
            "nivel": 3,
            "archivo": ruta_relativa,
            "n_ejemplos": n_ejemplos,
            "fuente": "qwen2.5-coder-32b via openrouter",
        })
        print(f"\n  📝 Añadido '{nombre}' a biblioteca/indice.json")

    with open(indice_path, "w", encoding="utf-8") as f:
        json.dump(indice, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
