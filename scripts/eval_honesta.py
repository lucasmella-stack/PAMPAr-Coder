#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
eval_honesta.py — Evaluación de generalización real (anti-memorización).

Diferencias vs benchmark.py:
  - Prompts 100% nuevos, no presentes en datos de entrenamiento ni en benchmark.py
  - Ejecuta el código generado y verifica resultados con asserts reales
  - Muestra el código generado completo para inspección visual
  - Veredicto claro: PASA / FALLA / SINTAXIS para cada caso

Uso:
  python -X utf8 scripts/eval_honesta.py --checkpoint checkpoints/pampar_v2_best.pt
  python -X utf8 scripts/eval_honesta.py --checkpoint checkpoints/pampar_v2_best.pt --temp 0.2
  python -X utf8 scripts/eval_honesta.py --checkpoint checkpoints/pampar_v2_best.pt --verbose
"""

import argparse
import ast
import sys
import textwrap
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# CASOS DE PRUEBA — nunca vistos por el modelo
# Cada caso: prompt, verificador (función que recibe el namespace ejecutado),
#             descripción legible y nivel de dificultad.
# =============================================================================

CASOS = [
    # ── Nivel 1: básicos ────────────────────────────────────────────────────
    {
        "nivel": 1,
        "desc": "Contar vocales en string",
        "prompt": "def contar_vocales(texto):\n    ",
        "verificar": lambda ns: (
            ns["contar_vocales"]("hola mundo") == 4
            and ns["contar_vocales"]("") == 0
            and ns["contar_vocales"]("xyz") == 0
        ),
    },
    {
        "nivel": 1,
        "desc": "Sumar dígitos de un número",
        "prompt": "def suma_digitos(n):\n    ",
        "verificar": lambda ns: (
            ns["suma_digitos"](123) == 6
            and ns["suma_digitos"](0) == 0
            and ns["suma_digitos"](999) == 27
        ),
    },
    {
        "nivel": 1,
        "desc": "Palíndromo",
        "prompt": "def es_palindromo(s):\n    ",
        "verificar": lambda ns: (
            ns["es_palindromo"]("racecar") is True
            and ns["es_palindromo"]("hello") is False
            and ns["es_palindromo"]("a") is True
        ),
    },
    # ── Nivel 2: listas/dicts ───────────────────────────────────────────────
    {
        "nivel": 2,
        "desc": "Aplanar lista anidada un nivel",
        "prompt": "def aplanar(lista):\n    ",
        "verificar": lambda ns: (
            ns["aplanar"]([[1, 2], [3, 4], [5]]) == [1, 2, 3, 4, 5]
            and ns["aplanar"]([]) == []
        ),
    },
    {
        "nivel": 2,
        "desc": "Frecuencia de elementos",
        "prompt": "def frecuencia(lista):\n    ",
        "verificar": lambda ns: (
            ns["frecuencia"](["a", "b", "a", "c", "b", "a"]) == {"a": 3, "b": 2, "c": 1}
            and ns["frecuencia"]([]) == {}
        ),
    },
    {
        "nivel": 2,
        "desc": "Segundo máximo de lista",
        "prompt": "def segundo_maximo(lista):\n    ",
        "verificar": lambda ns: (
            ns["segundo_maximo"]([3, 1, 4, 1, 5, 9, 2, 6]) == 6
            and ns["segundo_maximo"]([1, 2]) == 1
        ),
    },
    # ── Nivel 3: algoritmos ─────────────────────────────────────────────────
    {
        "nivel": 3,
        "desc": "Fibonacci iterativo (lista)",
        "prompt": "def fibonacci_lista(n):\n    \"\"\"Retorna los primeros n números de Fibonacci.\"\"\"\n    ",
        "verificar": lambda ns: (
            ns["fibonacci_lista"](7) == [0, 1, 1, 2, 3, 5, 8]
            and ns["fibonacci_lista"](1) == [0]
        ),
    },
    {
        "nivel": 3,
        "desc": "Anagramas",
        "prompt": "def son_anagramas(s1, s2):\n    ",
        "verificar": lambda ns: (
            ns["son_anagramas"]("listen", "silent") is True
            and ns["son_anagramas"]("hello", "world") is False
            and ns["son_anagramas"]("abc", "cab") is True
        ),
    },
    {
        "nivel": 3,
        "desc": "Rotar lista k posiciones",
        "prompt": "def rotar(lista, k):\n    ",
        "verificar": lambda ns: (
            ns["rotar"]([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3]
            and ns["rotar"]([1, 2, 3], 0) == [1, 2, 3]
        ),
    },
    # ── Nivel 4: OOP / patrones ─────────────────────────────────────────────
    {
        "nivel": 4,
        "desc": "Clase Pila con límite",
        "prompt": (
            "class PilaLimitada:\n"
            "    def __init__(self, limite):\n"
            "        self.limite = limite\n"
            "        self.items = []\n\n"
            "    def push(self, item):\n"
            "        "
        ),
        "verificar": lambda ns: (
            _test_pila_limitada(ns["PilaLimitada"])
        ),
    },
    {
        "nivel": 4,
        "desc": "Decorador que cachea resultados",
        "prompt": (
            "def cache_simple(func):\n"
            "    \"\"\"Decorador que memoriza resultados de llamadas anteriores.\"\"\"\n"
            "    memo = {}\n"
            "    def wrapper(*args):\n"
            "        "
        ),
        "verificar": lambda ns: _test_cache(ns["cache_simple"]),
    },
    # ── Nivel 5: typing / avanzado ──────────────────────────────────────────
    {
        "nivel": 5,
        "desc": "Generador de chunks",
        "prompt": (
            "from typing import Generator, TypeVar, List\n"
            "T = TypeVar('T')\n\n"
            "def chunks(lista: List[T], n: int) -> Generator[List[T], None, None]:\n"
            "    \"\"\"Divide lista en grupos de n elementos.\"\"\"\n"
            "    "
        ),
        "verificar": lambda ns: (
            list(ns["chunks"]([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
            and list(ns["chunks"]([], 3)) == []
        ),
    },
]


# Helpers de verificación (los lambdas no permiten bloques)
def _test_pila_limitada(PilaLimitada):
    p = PilaLimitada(2)
    p.push(1)
    p.push(2)
    try:
        p.push(3)  # debe lanzar o ignorar
    except Exception:
        pass
    return len(p.items) == 2


def _test_cache(cache_simple):
    llamadas = []

    @cache_simple
    def cuadrado(n):
        llamadas.append(n)
        return n * n

    assert cuadrado(4) == 16
    assert cuadrado(4) == 16  # segunda llamada, no debe recalcular
    assert len(llamadas) == 1, f"Se esperaba 1 llamada real, hubo {len(llamadas)}"
    assert cuadrado(5) == 25
    return True


# =============================================================================
# Cargadores (copiados de benchmark.py para no crear dependencia)
# =============================================================================

def cargar_modelo(checkpoint: Path, device: torch.device):
    from pampar.coder.v2.modelo import PampaRCoderV2
    from pampar.coder.v2.config import ConfigV2, PRESET_4GB, PRESET_8GB, PRESET_24GB, PRESET_1_5B
    import dataclasses

    PRESET_MAP = {"4GB": PRESET_4GB, "8GB": PRESET_8GB, "24GB": PRESET_24GB, "1_5B": PRESET_1_5B}
    config = PRESET_4GB

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_cfg = ckpt.get("config", {})
    state = ckpt.get("modelo", ckpt.get("model", ckpt))

    if isinstance(raw_cfg, ConfigV2):
        config = raw_cfg
    elif isinstance(raw_cfg, dict):
        preset = raw_cfg.get("preset")
        if preset in PRESET_MAP:
            config = PRESET_MAP[preset]

    emb = state.get("tok_emb.weight")
    if emb is not None and emb.shape != (config.vocab_size, config.dim):
        n_capas = sum(1 for k in state if k.startswith("capas.") and k.endswith(".ln1.weight"))
        config = ConfigV2(vocab_size=int(emb.shape[0]), dim=int(emb.shape[1]), n_capas=n_capas or config.n_capas)

    modelo = PampaRCoderV2(config).to(device)
    modelo.load_state_dict(state, strict=False)
    modelo.eval()
    return modelo, config


def cargar_tokenizer(vocab_size: int):
    import sentencepiece as spm
    path = (
        Path("data/tokenizer/code_tokenizer.model")
        if vocab_size == 16000
        else Path("data/tokenizer/pampar_48k.model")
    )
    if not path.exists():
        for p in [Path("data/tokenizer/code_tokenizer.model"), Path("data/tokenizer/pampar_48k.model")]:
            if p.exists():
                path = p
                break
    tok = spm.SentencePieceProcessor()
    tok.Load(str(path))
    return tok


# =============================================================================
# Generación
# =============================================================================

@torch.no_grad()
def generar(modelo, tokenizer, prompt: str, device, max_tokens=200, temperature=0.3) -> str:
    ids = tokenizer.Encode(prompt)
    generados = list(ids)

    for _ in range(max_tokens):
        ctx = torch.tensor([generados[-512:]], device=device)
        logits, _, _ = modelo(ctx)
        next_logits = logits[0, -1] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generados.append(next_token)

        decoded = tokenizer.Decode(generados[len(ids):])
        # Parar cuando el bloque está completo
        lines = decoded.split("\n")
        # Si hay una línea no indentada después de contenido, parar
        if len(lines) > 2:
            for i, line in enumerate(lines[1:], 1):
                if line and not line[0].isspace() and line.strip() not in ("", "pass"):
                    generados = generados[:len(ids) + len(tokenizer.Encode("\n".join(lines[:i])))]
                    return tokenizer.Decode(generados)
        if "\n\n" in decoded and len(decoded) > 30:
            break

    return tokenizer.Decode(generados)


# =============================================================================
# Ejecución segura del código generado
# =============================================================================

def ejecutar_y_verificar(codigo: str, verificador) -> tuple[str, str]:
    """
    Returns: (estado, detalle)
    estado: "PASA" | "FALLA" | "SINTAXIS" | "ERROR"
    """
    # 1. Validar sintaxis
    try:
        ast.parse(codigo)
    except SyntaxError as e:
        return "SINTAXIS", str(e)

    # 2. Ejecutar en namespace aislado
    ns = {}
    try:
        exec(compile(codigo, "<generated>", "exec"), ns)
    except Exception as e:
        return "ERROR_EXEC", f"{type(e).__name__}: {e}"

    # 3. Verificar resultados
    try:
        resultado = verificador(ns)
        if resultado:
            return "PASA", ""
        else:
            return "FALLA", "El verificador retornó False"
    except Exception as e:
        return "FALLA", f"{type(e).__name__}: {e}"


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluación honesta de PAMPAr")
    parser.add_argument("--checkpoint", default="checkpoints/pampar_v2_best.pt")
    parser.add_argument("--temp", type=float, default=0.3, help="Temperatura de generación")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--verbose", action="store_true", help="Mostrar código generado completo")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint)

    print(f"\n{'═'*60}")
    print(f"  EVAL HONESTA — PAMPAr Generalización")
    print(f"  Checkpoint : {checkpoint.name}")
    print(f"  Device     : {device} | Temp: {args.temp}")
    print(f"  Prompts    : {len(CASOS)} nuevos (no en dataset/benchmark)")
    print(f"{'═'*60}\n")

    print("  Cargando modelo...", end=" ", flush=True)
    modelo, config = cargar_modelo(checkpoint, device)
    tokenizer = cargar_tokenizer(config.vocab_size)
    n_params = sum(p.numel() for p in modelo.parameters()) / 1e6
    print(f"OK ({n_params:.0f}M params, vocab={config.vocab_size:,})\n")

    resultados = []
    t0 = time.time()

    for i, caso in enumerate(CASOS, 1):
        print(f"  [{i:02d}/{len(CASOS)}] Nivel {caso['nivel']} — {caso['desc']}")

        codigo = generar(
            modelo, tokenizer, caso["prompt"],
            device, args.max_tokens, args.temp
        )

        estado, detalle = ejecutar_y_verificar(codigo, caso["verificar"])

        icono = {"PASA": "✅", "FALLA": "❌", "SINTAXIS": "⚠️ ", "ERROR_EXEC": "💥"}.get(estado, "?")
        print(f"         {icono} {estado}" + (f" — {detalle}" if detalle else ""))

        if args.verbose or estado != "PASA":
            # Mostrar el código generado (indentado para legibilidad)
            print()
            for line in codigo.splitlines():
                print(f"    {line}")
            print()

        resultados.append({"desc": caso["desc"], "nivel": caso["nivel"], "estado": estado})

    # ── Resumen ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    pasan = sum(1 for r in resultados if r["estado"] == "PASA")
    fallan = sum(1 for r in resultados if r["estado"] == "FALLA")
    sintaxis = sum(1 for r in resultados if r["estado"] == "SINTAXIS")
    errores = sum(1 for r in resultados if r["estado"] == "ERROR_EXEC")
    total = len(resultados)

    por_nivel = {}
    for r in resultados:
        n = r["nivel"]
        por_nivel.setdefault(n, {"pasan": 0, "total": 0})
        por_nivel[n]["total"] += 1
        if r["estado"] == "PASA":
            por_nivel[n]["pasan"] += 1

    print(f"\n{'═'*60}")
    print(f"  RESULTADO FINAL — {elapsed:.1f}s")
    print(f"{'═'*60}")
    print(f"  ✅ Pasan      : {pasan}/{total}  ({pasan/total*100:.0f}%)")
    print(f"  ❌ Fallan     : {fallan}/{total}")
    print(f"  ⚠️  Sintaxis   : {sintaxis}/{total}")
    print(f"  💥 Error exec : {errores}/{total}")
    print()
    print("  Por nivel:")
    for nivel in sorted(por_nivel):
        d = por_nivel[nivel]
        barra = "█" * d["pasan"] + "░" * (d["total"] - d["pasan"])
        print(f"    Nivel {nivel}: {barra}  {d['pasan']}/{d['total']}")

    # Veredicto
    print()
    pct = pasan / total * 100
    if pct >= 80:
        veredicto = "🟢 GENERALIZA BIEN — el entrenamiento produjo conocimiento real"
    elif pct >= 50:
        veredicto = "🟡 GENERALIZACIÓN PARCIAL — aprende patrones pero falla en casos nuevos"
    elif pct >= 25:
        veredicto = "🟠 MEMORIZACIÓN PROBABLE — mejora en benchmark pero no generaliza"
    else:
        veredicto = "🔴 NO GENERALIZA — el modelo memorizó sin aprender"

    print(f"  {veredicto}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
