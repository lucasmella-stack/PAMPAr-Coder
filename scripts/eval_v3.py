#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
eval_v3.py — Evaluación de generalización real de PamparV3.

Ejecuta prompts nunca vistos, genera código y lo ejecuta con asserts reales.

Uso:
  python -X utf8 scripts/eval_v3.py
  python -X utf8 scripts/eval_v3.py --checkpoint checkpoints/v3_train.pt --temp 0.4
  python -X utf8 scripts/eval_v3.py --verbose
"""

import argparse
import ast
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# CASOS DE PRUEBA — nunca vistos por el modelo
# =============================================================================

CASOS = [
    # ── Nivel 1: básicos ────────────────────────────────────────────────────
    {
        "nivel": 1,
        "desc": "Contar vocales",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `contar_vocales(texto)` that returns the number "
            "of vowels (a, e, i, o, u, case-insensitive) in the string.\n"
            "### Solution:\n"
        ),
        "verificar": lambda ns: (
            ns["contar_vocales"]("hola mundo") == 4
            and ns["contar_vocales"]("") == 0
            and ns["contar_vocales"]("xyz") == 0
        ),
    },
    {
        "nivel": 1,
        "desc": "Sumar dígitos",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `suma_digitos(n)` that returns the sum of all "
            "digits of the non-negative integer n.\n"
            "### Solution:\n"
        ),
        "verificar": lambda ns: (
            ns["suma_digitos"](123) == 6
            and ns["suma_digitos"](0) == 0
            and ns["suma_digitos"](999) == 27
        ),
    },
    {
        "nivel": 1,
        "desc": "Palíndromo",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `es_palindromo(s)` that returns True if the "
            "string is a palindrome, False otherwise.\n"
            "### Solution:\n"
        ),
        "verificar": lambda ns: (
            ns["es_palindromo"]("racecar") is True
            and ns["es_palindromo"]("hello") is False
            and ns["es_palindromo"]("a") is True
        ),
    },
    {
        "nivel": 1,
        "desc": "Máximo de lista",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `maximo(lista)` that returns the maximum element "
            "of a non-empty list without using the built-in max().\n"
            "### Solution:\n"
        ),
        "verificar": lambda ns: (
            ns["maximo"]([3, 1, 4, 1, 5, 9]) == 9
            and ns["maximo"]([0]) == 0
            and ns["maximo"]([-1, -5, -2]) == -1
        ),
    },
    {
        "nivel": 1,
        "desc": "FizzBuzz single",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `fizzbuzz(n)` that returns 'FizzBuzz' if n is "
            "divisible by both 3 and 5, 'Fizz' if divisible by 3, 'Buzz' if divisible "
            "by 5, or the string representation of n otherwise.\n"
            "### Solution:\n"
        ),
        "verificar": lambda ns: (
            ns["fizzbuzz"](15) == "FizzBuzz"
            and ns["fizzbuzz"](3) == "Fizz"
            and ns["fizzbuzz"](5) == "Buzz"
            and ns["fizzbuzz"](7) == "7"
        ),
    },
    # ── Nivel 2: listas/dicts ───────────────────────────────────────────────
    {
        "nivel": 2,
        "desc": "Aplanar lista un nivel",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `aplanar(lista)` that flattens a list of lists "
            "by one level and returns the result as a single list.\n"
            "### Solution:\n"
        ),
        "verificar": lambda ns: (
            ns["aplanar"]([[1, 2], [3, 4], [5]]) == [1, 2, 3, 4, 5]
            and ns["aplanar"]([]) == []
        ),
    },
    {
        "nivel": 2,
        "desc": "Frecuencia de elementos",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `frecuencia(lista)` that returns a dictionary "
            "mapping each element to its count in the list.\n"
            "### Solution:\n"
        ),
        "verificar": lambda ns: (
            ns["frecuencia"]([1, 2, 2, 3, 3, 3]) == {1: 1, 2: 2, 3: 3}
            and ns["frecuencia"]([]) == {}
        ),
    },
    {
        "nivel": 2,
        "desc": "Lista de cuadrados pares",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `cuadrados_pares(n)` that returns a list of "
            "squares of all even numbers from 2 to n inclusive.\n"
            "### Solution:\n"
        ),
        "verificar": lambda ns: (
            ns["cuadrados_pares"](6) == [4, 16, 36]
            and ns["cuadrados_pares"](1) == []
        ),
    },
    {
        "nivel": 2,
        "desc": "Invertir diccionario",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `invertir_dict(d)` that returns a new dictionary "
            "with keys and values swapped.\n"
            "### Solution:\n"
        ),
        "verificar": lambda ns: (
            ns["invertir_dict"]({"a": 1, "b": 2}) == {1: "a", 2: "b"}
            and ns["invertir_dict"]({}) == {}
        ),
    },
    # ── Nivel 3: algoritmos ─────────────────────────────────────────────────
    {
        "nivel": 3,
        "desc": "Fibonacci iterativo",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `fibonacci(n)` that returns the n-th Fibonacci "
            "number (0-indexed: fibonacci(0)=0, fibonacci(1)=1, fibonacci(7)=13).\n"
            "### Solution:\n"
        ),
        "verificar": lambda ns: (
            ns["fibonacci"](0) == 0
            and ns["fibonacci"](1) == 1
            and ns["fibonacci"](7) == 13
            and ns["fibonacci"](10) == 55
        ),
    },
    {
        "nivel": 3,
        "desc": "Busqueda binaria",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `busqueda_binaria(lista, objetivo)` that returns "
            "the index of the target in a sorted list, or -1 if not found.\n"
            "### Solution:\n"
        ),
        "verificar": lambda ns: (
            ns["busqueda_binaria"]([1, 3, 5, 7, 9], 5) == 2
            and ns["busqueda_binaria"]([1, 3, 5, 7, 9], 4) == -1
            and ns["busqueda_binaria"]([], 1) == -1
        ),
    },
    {
        "nivel": 3,
        "desc": "Merge sort",
        "prompt": (
            "### Problem:\n"
            "Write a Python function `merge_sort(lista)` that returns a new sorted "
            "list using the merge sort algorithm.\n"
            "### Solution:\n"
        ),
        "verificar": lambda ns: (
            ns["merge_sort"]([3, 1, 4, 1, 5, 9, 2, 6]) == [1, 1, 2, 3, 4, 5, 6, 9]
            and ns["merge_sort"]([]) == []
            and ns["merge_sort"]([1]) == [1]
        ),
    },
    # ── Nivel 4: clases/OOP ─────────────────────────────────────────────────
    {
        "nivel": 4,
        "desc": "Clase Stack básica",
        "prompt": (
            "### Problem:\n"
            "Write a Python class `Stack` with methods `push(item)` and `pop()` "
            "implementing a LIFO stack.\n"
            "### Solution:\n"
        ),
        "verificar": lambda ns: (
            (s := ns["Stack"]()) is not None
            and (s.push(1) or True)
            and (s.push(2) or True)
            and s.pop() == 2
            and s.pop() == 1
        ),
    },
    {
        "nivel": 4,
        "desc": "Clase Punto con distancia",
        "prompt": (
            "### Problem:\n"
            "Write a Python class `Punto` with attributes `x` and `y`, and a method "
            "`distancia(otro)` that returns the Euclidean distance to another Punto.\n"
            "### Solution:\n"
        ),
        "verificar": lambda ns: (
            ns["Punto"](0, 0).distancia(ns["Punto"](3, 4)) == 5.0
        ),
    },
    # ── Nivel 5: funcional/avanzado ─────────────────────────────────────────
    {
        "nivel": 5,
        "desc": "Memoización con decorador",
        "prompt": (
            "### Problem:\n"
            "Write a Python higher-order function `memoize(fn)` that returns a wrapped "
            "version of fn that caches results by argument.\n"
            "### Solution:\n"
        ),
        "verificar": lambda ns: (
            (fn := ns["memoize"](lambda x: x * 2)) is not None
            and fn(5) == 10
            and fn(5) == 10  # cache hit
        ),
    },
    {
        "nivel": 5,
        "desc": "Generador de números primos",
        "prompt": (
            "### Problem:\n"
            "Write a Python generator function `primos_hasta(n)` that yields all "
            "prime numbers up to and including n.\n"
            "### Solution:\n"
        ),
        "verificar": lambda ns: (
            list(ns["primos_hasta"](20)) == [2, 3, 5, 7, 11, 13, 17, 19]
        ),
    },
]


# =============================================================================
# Carga del modelo v3
# =============================================================================

def cargar_modelo_v3(checkpoint: Path, device: torch.device):
    import dataclasses
    from pampar.coder.v3.modelo import PamparV3
    from pampar.coder.v3.config import ConfigV3, PRESET_V3

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_cfg = ckpt.get("config", {})
    state = ckpt.get("modelo", ckpt)

    # Reconstruir ConfigV3 filtrando solo los campos válidos del dataclass
    if isinstance(raw_cfg, dict) and "dim" in raw_cfg:
        campos_validos = {f.name for f in dataclasses.fields(ConfigV3)}
        kwargs = {k: v for k, v in raw_cfg.items() if k in campos_validos}
        try:
            config = ConfigV3(**kwargs)
        except TypeError:
            config = PRESET_V3
    elif isinstance(raw_cfg, ConfigV3):
        config = raw_cfg
    else:
        config = PRESET_V3

    modelo = PamparV3(config).to(device)
    missing, unexpected = modelo.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] keys faltantes: {len(missing)}")
    modelo.eval()
    return modelo, config


def cargar_tokenizer(vocab_size: int = 48000):
    import sentencepiece as spm
    candidates = [
        Path("data/tokenizer/pampar_48k.model"),
        Path("data/tokenizer/code_tokenizer.model"),
    ]
    for p in candidates:
        if p.exists():
            tok = spm.SentencePieceProcessor()
            tok.Load(str(p))
            return tok
    raise FileNotFoundError("No se encontró el tokenizer")


# =============================================================================
# Extracción de firma para modo guiado
# =============================================================================

def extraer_firma(prompt: str) -> str:
    """Extract function/class signature from prompt for guided generation."""
    m = re.search(r'class `(\w+)`', prompt)
    if m:
        return f"class {m.group(1)}:"
    m = re.search(r'function `(\w+\([^)]*\))`', prompt)
    if m:
        return f"def {m.group(1)}:"
    return ""


# =============================================================================
# Generación greedy / top-p
# =============================================================================

@torch.no_grad()
def generar(modelo, tokenizer, prompt: str, device, max_tokens: int = 384,
           temperature: float = 0.1, repetition_penalty: float = 1.2) -> str:
    ids = tokenizer.Encode(prompt)
    generados = list(ids)

    for _ in range(max_tokens):
        ctx = torch.tensor([generados[-512:]], dtype=torch.long, device=device)
        logits, _, _ = modelo(ctx)
        next_logits = logits[0, -1]

        # Penalizar tokens ya generados para evitar degeneración
        if repetition_penalty != 1.0:
            seen = set(generados[len(ids):])
            for token_id in seen:
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= repetition_penalty
                else:
                    next_logits[token_id] *= repetition_penalty

        if temperature <= 0.0:
            next_token = int(next_logits.argmax())
        else:
            next_logits = next_logits / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = int(torch.multinomial(probs, 1))

        generados.append(next_token)
        decoded = tokenizer.Decode(generados[len(ids):]).replace('\u2047', '\n')

        # Parar si el modelo empieza una nueva sección (formato instrucción)
        if "###" in decoded:
            idx = decoded.index("###")
            if idx > 10:  # al menos algo de código generado
                return prompt + decoded[:idx].rstrip()

        # Parar cuando termina la función/clase (línea sin sangría después de contenido)
        lines = decoded.split("\n")
        if len(lines) > 3:
            for i, line in enumerate(lines[2:], 2):
                if line and not line[0].isspace() and line.strip() not in ("", "pass"):
                    partial = "\n".join(lines[:i])
                    return prompt + partial

        if "\n\n" in decoded and len(decoded) > 20:
            break

    return tokenizer.Decode(generados).replace('\u2047', '\n')


# =============================================================================
# Normalización de indentación
# =============================================================================

def _normalizar_indentacion(codigo: str) -> str:
    """Corregir indentación inconsistente (ej. 5 espacios → 4) redondeando a múltiplos de 4."""
    lines = codigo.split('\n')
    if not lines:
        return codigo

    fixed = [lines[0]]  # Primera línea (def/class) se mantiene
    for line in lines[1:]:
        stripped = line.lstrip()
        if not stripped:
            fixed.append('')
            continue
        spaces = len(line) - len(stripped)
        # Redondear a múltiplo de 4 más cercano, mínimo 4 si dentro de función/clase
        normalized = round(spaces / 4) * 4
        if normalized < 4 and lines[0].lstrip().startswith(('def ', 'class ')):
            normalized = 4
        fixed.append(' ' * normalized + stripped)

    return '\n'.join(fixed)


# =============================================================================
# Ejecución segura
# =============================================================================

def ejecutar_y_verificar(codigo: str, verificador) -> tuple[str, str]:
    import textwrap

    # Si el output es formato instrucción, extraer solo el código después de ### Solution:
    if "### Solution:" in codigo:
        codigo = codigo.split("### Solution:")[-1].lstrip("\n")

    # Extraer código dentro de ```python ... ``` si existe
    if "```python" in codigo:
        bloque = codigo.split("```python", 1)[1]
        if "```" in bloque:
            bloque = bloque.split("```", 1)[0]
        codigo = bloque
    elif "```" in codigo:
        # Bloque sin tag de lenguaje
        bloque = codigo.split("```", 1)[1]
        if "```" in bloque:
            bloque = bloque.split("```", 1)[0]
        codigo = bloque

    # Normalizar indentación (dedent elimina espacios líderes comunes)
    codigo = textwrap.dedent(codigo).strip()

    try:
        ast.parse(codigo)
    except SyntaxError:
        # Intentar corregir indentación inconsistente (4 vs 5 espacios)
        codigo = _normalizar_indentacion(codigo)
        try:
            ast.parse(codigo)
        except SyntaxError as e:
            return "SINTAXIS", str(e)

    ns = {}
    try:
        exec(compile(codigo, "<generated>", "exec"), ns)
    except Exception as e:
        return "ERROR_EXEC", f"{type(e).__name__}: {e}"

    try:
        resultado = verificador(ns)
        return ("PASA", "") if resultado else ("FALLA", "verificador → False")
    except KeyError as e:
        return "FALLA", f"función no definida: {e}"
    except Exception as e:
        return "FALLA", f"{type(e).__name__}: {e}"


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/v3_train.pt")
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--rep-penalty", type=float, default=1.2)
    parser.add_argument("--guided", action="store_true",
                        help="Include function/class signature in prompt (HumanEval style)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cuda' o 'cpu'")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)

    print(f"\n{'═'*65}")
    print(f"  EVAL HONESTA — PamparV3 Generalización")
    print(f"  Checkpoint : {checkpoint.name}  ({checkpoint.stat().st_size/1e9:.2f} GB)")
    mode_str = "GUIDED" if args.guided else "OPEN"
    print(f"  Device     : {device}  |  Temp: {args.temp}  |  RepPen: {args.rep_penalty}")
    print(f"  Mode       : {mode_str}")
    print(f"  Prompts    : {len(CASOS)} (nunca vistos en entrenamiento)")
    print(f"{'═'*65}\n")

    print("  Cargando modelo...", end=" ", flush=True)
    t0 = time.time()
    modelo, config = cargar_modelo_v3(checkpoint, device)
    tokenizer = cargar_tokenizer(config.vocab_size)
    n_params = sum(p.numel() for p in modelo.parameters()) / 1e6
    print(f"OK ({n_params:.1f}M params, vocab={config.vocab_size:,}, {time.time()-t0:.1f}s)\n")

    resultados = []
    t_start = time.time()

    for i, caso in enumerate(CASOS, 1):
        print(f"  [{i:02d}/{len(CASOS)}] Nivel {caso['nivel']} — {caso['desc']}", end="  ", flush=True)

        t_gen = time.time()
        prompt_gen = caso["prompt"]
        if args.guided:
            firma = extraer_firma(caso["prompt"])
            if firma:
                # Incluir hint de indentación (4 espacios) para primar al modelo
                prompt_gen = caso["prompt"] + "```python\n" + firma + "\n    "
        codigo = generar(
            modelo, tokenizer, prompt_gen,
            device, args.max_tokens, args.temp, args.rep_penalty
        )
        dt = time.time() - t_gen

        estado, detalle = ejecutar_y_verificar(codigo, caso["verificar"])

        ICONOS = {"PASA": "✅", "FALLA": "❌", "SINTAXIS": "⚠️", "ERROR_EXEC": "💥"}
        icono = ICONOS.get(estado, "?")
        print(f"[{dt:.1f}s] {icono} {estado}" + (f" — {detalle}" if detalle else ""))

        if args.verbose or estado != "PASA":
            print()
            for line in codigo.splitlines():
                print(f"    {line}")
            print()

        resultados.append({"desc": caso["desc"], "nivel": caso["nivel"], "estado": estado})

    # ── Resumen ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    pasan   = sum(1 for r in resultados if r["estado"] == "PASA")
    fallan  = sum(1 for r in resultados if r["estado"] == "FALLA")
    sintax  = sum(1 for r in resultados if r["estado"] == "SINTAXIS")
    errores = sum(1 for r in resultados if r["estado"] == "ERROR_EXEC")
    total   = len(resultados)

    por_nivel: dict = {}
    for r in resultados:
        n = r["nivel"]
        por_nivel.setdefault(n, {"pasan": 0, "total": 0})
        por_nivel[n]["total"] += 1
        if r["estado"] == "PASA":
            por_nivel[n]["pasan"] += 1

    print(f"\n{'═'*65}")
    print(f"  RESULTADO FINAL — {elapsed:.0f}s total")
    print(f"{'═'*65}")
    print(f"  ✅ Pasan      : {pasan}/{total}  ({pasan/total*100:.0f}%)")
    print(f"  ❌ Fallan     : {fallan}/{total}")
    print(f"  ⚠️  Sintaxis   : {sintax}/{total}")
    print(f"  💥 Error exec : {errores}/{total}")
    print()
    print("  Por nivel:")
    for nivel in sorted(por_nivel):
        d = por_nivel[nivel]
        barra = "█" * d["pasan"] + "░" * (d["total"] - d["pasan"])
        print(f"    Nivel {nivel}: {barra}  {d['pasan']}/{d['total']}")

    print()
    pct = pasan / total * 100
    if pct >= 80:
        veredicto = "🟢 GENERALIZA BIEN — el modelo aprendió de verdad"
    elif pct >= 50:
        veredicto = "🟡 PARCIAL — aprende patrones pero falla en casos nuevos"
    elif pct >= 25:
        veredicto = "🟠 PROBABLE MEMORIZACIÓN — mejora en benchmark pero no generaliza"
    else:
        veredicto = "🔴 NO GENERALIZA — 134k pasos no fueron suficientes"

    print(f"  {veredicto}")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
