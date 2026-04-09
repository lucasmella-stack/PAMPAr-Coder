# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr Brain Scanner — Visualización y diagnóstico de la arquitectura cerebral 2D.

Muestra cómo PamparV3 procesa tokens internamente:
  - Activaciones territoriales (Tálamo routing por stream)
  - Evolución por nivel (cómo cambian las activaciones a través de 5 niveles)
  - Fibras blancas (LateralGate scale — comunicación entre streams)
  - Zonas de Brodmann activas por token
  - Early Exit (qué nivel puede salir antes)
  - Distribución de pesos por componente
  - Precisión de routing vs LLAVES (ground truth)
  - Margen de decisión de routing (ambigüedad)
  - Suite de tests con métricas agregadas
  - Comparación de checkpoints
  - Generación de código + evaluación

Uso:
  python scripts/brain_scanner.py --code "def fibonacci(n):"
  python scripts/brain_scanner.py --suite
  python scripts/brain_scanner.py --compare ckpt_a.pt ckpt_b.pt
  python scripts/brain_scanner.py --generate "def factorial(n):"
  python scripts/brain_scanner.py --weights
  python scripts/brain_scanner.py --code "x = [i**2 for i in range(5)]" --html scan.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# Agregar raíz del proyecto al path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pampar.coder.v3.config import PRESET_V3, ConfigV3
from pampar.coder.v3.llaves import clasificar_token
from pampar.coder.v3.modelo import PamparV3
from pampar.coder.v3.talamo import TalamoInicial
from pampar.coder.v3.zonas import ZONA_TERRITORIO, Territorio, Zona

# =============================================================================
# CONSTANTES
# =============================================================================

STREAM_NAMES = ["SINTAXIS", "SEMANTICA", "LOGICO", "ESTRUCTURAL"]
STREAM_COLORS = [
    "\033[94m",
    "\033[92m",
    "\033[93m",
    "\033[95m",
]  # blue, green, yellow, purple
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

# Bloques para barras
BLOCKS = " ▏▎▍▌▋▊▉█"

# Nombres de zonas abreviados
ZONA_SHORT = {z: z.name.replace("B", "").replace("_", " ") for z in Zona}

# Suite de código diverso para test comprehensivo
CODE_SUITE = [
    # Keywords de control
    ("keywords", "def fibonacci(n):"),
    ("clase", "class DataProcessor:"),
    ("imports", "from pathlib import Path"),
    ("loop", "for i in range(10):"),
    ("condicional", "if x > 0 and y < 10:"),
    ("excepcion", "try:\n    result = 1 / 0\nexcept ZeroDivisionError:"),
    ("async", "async def fetch(url):"),
    # Operadores y lógica
    ("aritmetica", "result = a + b * c - d / e"),
    ("comparacion", "x == y or x != z"),
    ("asignacion", "total += price * quantity"),
    # Semántica (ids, literals, tipos)
    ("literals", "name = 'hello world'"),
    ("numeros", "pi = 3.14159"),
    ("tipos", "items: list[int] = []"),
    ("builtins", "print(len(range(10)))"),
    ("magic", "def __init__(self, value):"),
    # Estructural
    ("comprehension", "squares = [x**2 for x in range(10)]"),
    ("lambda", "fn = lambda x: x * 2"),
    ("decorador", "@staticmethod\ndef create():"),
    ("return", "return sorted(data, key=lambda x: x.name)"),
    ("with", "with open('file.txt') as f:"),
]


# =============================================================================
# CARGA DEL MODELO (delegada a pampar.inference)
# =============================================================================

from pampar.inference import load_model

# =============================================================================
# BARRA VISUAL
# =============================================================================


def barra(valor: float, ancho: int = 20, color: str = "") -> str:
    """Dibuja una barra horizontal proporcional al valor [0, 1]."""
    v = max(0.0, min(1.0, valor))
    lleno = int(v * ancho)
    frac = int((v * ancho - lleno) * 8)
    chars = "█" * lleno
    if frac > 0 and lleno < ancho:
        chars += BLOCKS[frac]
        lleno += 1
    chars += " " * (ancho - lleno)
    pct = f"{v * 100:5.1f}%"
    if color:
        return f"{color}{chars}{RESET} {pct}"
    return f"{chars} {pct}"


def heatmap_char(valor: float) -> str:
    """Devuelve un caracter coloreado para heatmap (0=azul, 1=rojo)."""
    v = max(0.0, min(1.0, valor))
    if v < 0.2:
        return f"\033[34m░{RESET}"  # azul
    if v < 0.4:
        return f"\033[36m▒{RESET}"  # cyan
    if v < 0.6:
        return f"\033[32m▓{RESET}"  # verde
    if v < 0.8:
        return f"\033[33m█{RESET}"  # amarillo
    return f"\033[31m█{RESET}"  # rojo


# =============================================================================
# FORWARD INSTRUMENTADO
# =============================================================================


@torch.no_grad()
def forward_instrumentado(
    model: PamparV3,
    input_ids: torch.Tensor,
) -> dict:
    """
    Ejecuta el forward capturando todas las activaciones internas.

    Returns:
        dict con:
          tokens:         list[str] — tokens decodificados
          zona_acts:      [L, 52]  — activaciones de zona (Tálamo)
          terr_por_nivel: list[[L, 4]] — activaciones de territorio por nivel
          confianza:      list[float] — confianza de early exit por nivel
          lateral_scales: [n_levels, 4] — escalas de LateralGate
          stream_norms:   [n_levels, 4] — norma L2 de cada stream por nivel
          attn_norms:     [n_levels] — norma del output de atención por nivel
    """
    config = model.config
    B, L = input_ids.shape

    # 1. Embedding
    x = model.emb_drop(model.tok_emb(input_ids))

    # 2. Tálamo inicial
    terr_acts, zona_acts = model.talamo(x, input_ids)

    # 3. Inicializar streams
    streams = [x.clone() for _ in range(config.n_streams)]

    # Coleccionar datos por nivel
    terr_por_nivel = [terr_acts[0].cpu()]  # nivel 0 = entrada
    confianzas = []
    lateral_scales = []
    stream_norms = []
    attn_norms = []

    # 4. Pasar por cada nivel
    for i, nivel in enumerate(model.niveles):
        # --- Capturar escalas de LateralGate ANTES del forward ---
        scales = nivel.lateral.scale.detach().cpu().tolist()
        lateral_scales.append(scales)

        # --- Forward del nivel ---
        # Reproducimos el forward manualmente para capturar intermedios

        # 4a. Representación combinada
        x_combined = sum(
            streams[t] * terr_acts[:, :, t : t + 1] for t in range(config.n_streams)
        )

        # 4b. Atención compartida
        x_attn = nivel.drop(nivel.attn(nivel.norm_attn(x_combined)))
        attn_norms.append(x_attn[0].norm(dim=-1).mean().item())

        # 4c. Re-routing
        terr_acts = nivel.talamo_nivel(
            x_combined + x_attn, terr_acts, TalamoInicial.agregar_fn
        )

        # 4d. FFN por stream
        new_streams = []
        for t in range(config.n_streams):
            h_normed = nivel.norm_streams[t](streams[t] + x_attn)
            h = nivel.ffns[t](h_normed) * terr_acts[:, :, t : t + 1]
            new_streams.append(streams[t] + nivel.drop(h))

        # 4e. Lateral gates
        streams = nivel.lateral(new_streams, terr_acts)

        # 4f. Confianza Early Exit
        x_out = sum(
            streams[t] * terr_acts[:, :, t : t + 1] for t in range(config.n_streams)
        )
        per_token_conf = torch.sigmoid(nivel.exit_head(x_out)).squeeze(-1)
        k = max(1, int(per_token_conf.numel() * config.exit_percentile))
        conf = per_token_conf.reshape(-1).topk(k, largest=False).values.mean().item()
        confianzas.append(conf)

        # Capturar datos por nivel
        terr_por_nivel.append(terr_acts[0].cpu())
        norms = [
            streams[t][0].norm(dim=-1).mean().item() for t in range(config.n_streams)
        ]
        stream_norms.append(norms)

    return {
        "zona_acts": zona_acts[0].cpu(),  # [L, 52]
        "terr_por_nivel": terr_por_nivel,  # list[[L, 4]]
        "confianza": confianzas,  # [n_levels]
        "lateral_scales": lateral_scales,  # [n_levels, 4]
        "stream_norms": stream_norms,  # [n_levels, 4]
        "attn_norms": attn_norms,  # [n_levels]
    }


# =============================================================================
# VISUALIZACIÓN: ACTIVACIONES TERRITORIALES
# =============================================================================


def mostrar_routing(tokens: list[str], info: dict) -> str:
    """Muestra cómo el Tálamo enruta cada token a los 4 streams."""
    lines = []
    lines.append(f"\n{BOLD}═══ TÁLAMO: ROUTING INICIAL ═══{RESET}\n")
    lines.append(
        f"  {'Token':<15} {'SINTAXIS':>10} {'SEMANTICA':>10} {'LOGICO':>10} {'ESTRUCTURAL':>10}  Dominante"
    )
    lines.append(f"  {'─' * 15} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 13} {'─' * 12}")

    terr_0 = info["terr_por_nivel"][0]  # [L, 4]

    for i, tok in enumerate(tokens):
        acts = terr_0[i].tolist()
        dominant = max(range(4), key=lambda t: acts[t])
        color = STREAM_COLORS[dominant]

        tok_display = repr(tok).strip("'")[:14]
        vals = " ".join(f"{a:10.3f}" for a in acts)
        lines.append(
            f"  {tok_display:<15} {vals}  {color}{STREAM_NAMES[dominant]}{RESET}"
        )

    return "\n".join(lines)


# =============================================================================
# VISUALIZACIÓN: ZONAS DE BRODMANN
# =============================================================================


def mostrar_zonas(tokens: list[str], info: dict) -> str:
    """Muestra las zonas de Brodmann más activas por token."""
    lines = []
    lines.append(f"\n{BOLD}═══ ZONAS DE BRODMANN ACTIVAS ═══{RESET}\n")

    zona_acts = info["zona_acts"]  # [L, 52]

    for i, tok in enumerate(tokens):
        acts = zona_acts[i]
        top5_idx = acts.topk(5).indices.tolist()
        top5_vals = acts.topk(5).values.tolist()

        tok_display = repr(tok).strip("'")[:12]
        zonas_str = "  ".join(
            f"{heatmap_char(v)}{list(Zona)[idx].name[3:]:<12}{v:.2f}"
            for idx, v in zip(top5_idx, top5_vals)
        )
        lines.append(f"  {tok_display:<14} {zonas_str}")

    return "\n".join(lines)


# =============================================================================
# VISUALIZACIÓN: EVOLUCIÓN POR NIVEL
# =============================================================================


def mostrar_evolucion(tokens: list[str], info: dict) -> str:
    """Muestra cómo evolucionan las activaciones territoriales a través de los 5 niveles."""
    lines = []
    lines.append(f"\n{BOLD}═══ EVOLUCIÓN POR NIVEL (profundidad cortical) ═══{RESET}\n")

    n_levels = len(info["confianza"])

    for t_idx, name in enumerate(STREAM_NAMES):
        color = STREAM_COLORS[t_idx]
        lines.append(f"  {color}{BOLD}{name}{RESET}")
        lines.append(
            f"  {'Token':<12} "
            + " ".join(f"{'N' + str(n):<8}" for n in range(n_levels + 1))
        )

        for i, tok in enumerate(tokens):
            tok_display = repr(tok).strip("'")[:11]
            vals = []
            for n in range(n_levels + 1):
                v = info["terr_por_nivel"][n][i, t_idx].item()
                vals.append(f"{heatmap_char(v)} {v:.2f} ")
            lines.append(f"  {tok_display:<12} " + " ".join(vals))
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# VISUALIZACIÓN: FIBRAS BLANCAS (LateralGate)
# =============================================================================


def mostrar_fibras_blancas(info: dict) -> str:
    """Muestra los pesos de comunicación lateral entre streams."""
    lines = []
    lines.append(f"\n{BOLD}═══ FIBRAS BLANCAS (LateralGate scales) ═══{RESET}")
    lines.append(f"  Escala aprendida de comunicación entre streams por nivel.\n")

    scales = info["lateral_scales"]  # [n_levels, 4]

    lines.append(
        f"  {'Nivel':<8} "
        + " ".join(
            f"{STREAM_COLORS[t]}{name:<14}{RESET}"
            for t, name in enumerate(STREAM_NAMES)
        )
    )
    lines.append(f"  {'─' * 8} " + " ".join("─" * 14 for _ in STREAM_NAMES))

    for n, level_scales in enumerate(scales):
        vals = " ".join(
            f"{STREAM_COLORS[t]}{barra(abs(s), 10)}{RESET}"
            for t, s in enumerate(level_scales)
        )
        lines.append(f"  Nivel {n:<3} {vals}")

    # Resumen: qué stream comunica más
    avg_scales = [
        sum(abs(scales[n][t]) for n in range(len(scales))) / len(scales)
        for t in range(4)
    ]
    max_idx = max(range(4), key=lambda t: avg_scales[t])
    lines.append(
        f"\n  Stream más comunicativo: {STREAM_COLORS[max_idx]}{BOLD}{STREAM_NAMES[max_idx]}{RESET} (escala promedio: {avg_scales[max_idx]:.4f})"
    )

    return "\n".join(lines)


# =============================================================================
# VISUALIZACIÓN: CONFIANZA EARLY EXIT
# =============================================================================


def mostrar_early_exit(info: dict) -> str:
    """Muestra la confianza de early exit por nivel."""
    lines = []
    lines.append(f"\n{BOLD}═══ EARLY EXIT (confianza por nivel) ═══{RESET}")
    lines.append(
        f"  Umbral: {PRESET_V3.umbral_exit:.0%} — mín {PRESET_V3.capas_min} niveles\n"
    )

    for n, conf in enumerate(info["confianza"]):
        color = "\033[32m" if conf >= PRESET_V3.umbral_exit else "\033[31m"
        marker = (
            " ◄ EXIT"
            if conf >= PRESET_V3.umbral_exit and n >= PRESET_V3.capas_min - 1
            else ""
        )
        lines.append(f"  Nivel {n}  {barra(conf, 30, color)}{BOLD}{marker}{RESET}")

    return "\n".join(lines)


# =============================================================================
# VISUALIZACIÓN: NORMAS DE STREAMS
# =============================================================================


def mostrar_stream_norms(info: dict) -> str:
    """Muestra la norma L2 de cada stream por nivel (actividad del stream)."""
    lines = []
    lines.append(f"\n{BOLD}═══ ACTIVIDAD DE STREAMS (norma L2 promedio) ═══{RESET}\n")

    norms = info["stream_norms"]  # [n_levels, 4]
    # Normalizar al max para visualización
    max_norm = max(max(level) for level in norms)

    lines.append(
        f"  {'Nivel':<8} "
        + " ".join(
            f"{STREAM_COLORS[t]}{name:<14}{RESET}"
            for t, name in enumerate(STREAM_NAMES)
        )
    )
    lines.append(f"  {'─' * 8} " + " ".join("─" * 14 for _ in STREAM_NAMES))

    for n, level_norms in enumerate(norms):
        vals = " ".join(
            f"{STREAM_COLORS[t]}{barra(v / max_norm, 10)}{RESET}"
            for t, v in enumerate(level_norms)
        )
        lines.append(f"  Nivel {n:<3} {vals}")

    return "\n".join(lines)


# =============================================================================
# TERRITORY TABLE (reutiliza lógica de neuro_trainer)
# =============================================================================


def _build_territory_table(tokenizer: object) -> torch.Tensor:
    """Construye lookup table: token_id → territorio target (0-3)."""
    vocab_size = tokenizer.GetPieceSize()
    table = torch.zeros(vocab_size, dtype=torch.long)
    for token_id in range(vocab_size):
        piece = tokenizer.IdToPiece(token_id)
        zona, _conf = clasificar_token(piece)
        table[token_id] = ZONA_TERRITORIO[zona].value
    return table


# =============================================================================
# VISUALIZACIÓN: PRECISIÓN DE ROUTING VS LLAVES
# =============================================================================


def mostrar_precision(
    tokens: list[str],
    token_ids: list[int],
    info: dict,
    territory_table: torch.Tensor,
) -> str:
    """Compara routing actual vs territorio esperado de LLAVES por token."""
    lines = []
    lines.append(f"\n{BOLD}═══ PRECISIÓN DE ROUTING vs LLAVES ═══{RESET}\n")
    lines.append(
        f"  {'Token':<15} {'Esperado':<14} {'Actual N0':<14} {'Actual N5':<14} {'N0':>3} {'N5':>3}  Zona LLAVES"
    )
    lines.append(
        f"  {'─' * 15} {'─' * 14} {'─' * 14} {'─' * 14} {'─' * 3} {'─' * 3}  {'─' * 20}"
    )

    n_levels = len(info["confianza"])
    terr_0 = info["terr_por_nivel"][0]
    terr_last = info["terr_por_nivel"][n_levels]

    correct_n0 = 0
    correct_nlast = 0
    total = len(tokens)

    for i, (tok, tid) in enumerate(zip(tokens, token_ids)):
        expected = territory_table[tid].item()
        actual_n0 = terr_0[i].argmax().item()
        actual_nlast = terr_last[i].argmax().item()

        # Clasificación LLAVES para mostrar la zona
        zona, conf = clasificar_token(tok)

        match_n0 = actual_n0 == expected
        match_nlast = actual_nlast == expected
        if match_n0:
            correct_n0 += 1
        if match_nlast:
            correct_nlast += 1

        sym_n0 = f"\033[32m✓{RESET}" if match_n0 else f"\033[31m✗{RESET}"
        sym_nlast = f"\033[32m✓{RESET}" if match_nlast else f"\033[31m✗{RESET}"
        exp_color = STREAM_COLORS[expected]
        act0_color = STREAM_COLORS[actual_n0]
        actL_color = STREAM_COLORS[actual_nlast]

        tok_display = repr(tok).strip("'")[:14]
        zona_str = f"{zona.name[3:]} ({conf:.0%})"
        lines.append(
            f"  {tok_display:<15} "
            f"{exp_color}{STREAM_NAMES[expected]:<14}{RESET}"
            f"{act0_color}{STREAM_NAMES[actual_n0]:<14}{RESET}"
            f"{actL_color}{STREAM_NAMES[actual_nlast]:<14}{RESET}"
            f" {sym_n0}   {sym_nlast}  {zona_str}"
        )

    acc_n0 = correct_n0 / total * 100 if total > 0 else 0
    acc_nlast = correct_nlast / total * 100 if total > 0 else 0
    color_n0 = (
        "\033[32m" if acc_n0 >= 80 else "\033[33m" if acc_n0 >= 50 else "\033[31m"
    )
    color_nlast = (
        "\033[32m" if acc_nlast >= 80 else "\033[33m" if acc_nlast >= 50 else "\033[31m"
    )

    lines.append(
        f"\n  {BOLD}Accuracy N0: {color_n0}{acc_n0:.1f}%{RESET}  ({correct_n0}/{total})"
    )
    lines.append(
        f"  {BOLD}Accuracy N{n_levels}: {color_nlast}{acc_nlast:.1f}%{RESET}  ({correct_nlast}/{total})"
    )

    return "\n".join(lines)


# =============================================================================
# VISUALIZACIÓN: MARGEN DE ROUTING
# =============================================================================


def mostrar_margen(tokens: list[str], info: dict) -> str:
    """Muestra el margen de decisión del routing (1er vs 2do stream)."""
    lines = []
    lines.append(f"\n{BOLD}═══ MARGEN DE ROUTING (confianza de decisión) ═══{RESET}")
    lines.append(f"  Margen = act(dominante) - act(segundo). Bajo = ambiguo.\n")

    n_levels = len(info["confianza"])
    terr_last = info["terr_por_nivel"][n_levels]

    lines.append(
        f"  {'Token':<15} {'Dominante':<12} {'1er':>6} {'2do':>6} {'Margen':>8}  Visual"
    )
    lines.append(f"  {'─' * 15} {'─' * 12} {'─' * 6} {'─' * 6} {'─' * 8}  {'─' * 20}")

    margins = []
    for i, tok in enumerate(tokens):
        acts = terr_last[i].tolist()
        sorted_acts = sorted(enumerate(acts), key=lambda x: x[1], reverse=True)
        dominant = sorted_acts[0]
        second = sorted_acts[1]
        margin = dominant[1] - second[1]
        margins.append(margin)

        color = STREAM_COLORS[dominant[0]]
        m_color = (
            "\033[32m" if margin > 0.05 else "\033[33m" if margin > 0.02 else "\033[31m"
        )

        tok_display = repr(tok).strip("'")[:14]
        lines.append(
            f"  {tok_display:<15} "
            f"{color}{STREAM_NAMES[dominant[0]]:<12}{RESET}"
            f"{dominant[1]:>6.3f} {second[1]:>6.3f} "
            f"{m_color}{margin:>8.4f}{RESET}  "
            f"{barra(min(1.0, margin * 10), 15, m_color)}"
        )

    avg_margin = sum(margins) / len(margins) if margins else 0
    min_margin = min(margins) if margins else 0
    m_color = (
        "\033[32m"
        if avg_margin > 0.05
        else "\033[33m"
        if avg_margin > 0.02
        else "\033[31m"
    )
    lines.append(f"\n  {BOLD}Margen promedio: {m_color}{avg_margin:.4f}{RESET}")
    lines.append(f"  {BOLD}Margen mínimo:  {m_color}{min_margin:.4f}{RESET}")
    if min_margin < 0.01:
        lines.append(
            f"  {BOLD}\033[31m⚠ Tokens con margen <0.01 → routing casi aleatorio{RESET}"
        )

    return "\n".join(lines)


# =============================================================================
# VISUALIZACIÓN: RESUMEN CUANTITATIVO
# =============================================================================


def mostrar_resumen(
    tokens: list[str],
    token_ids: list[int],
    info: dict,
    territory_table: torch.Tensor,
) -> str:
    """Panel de métricas agregadas para evaluación rápida."""
    lines = []
    lines.append(f"\n{BOLD}═══ RESUMEN DE SALUD DEL MODELO ═══{RESET}\n")

    n_levels = len(info["confianza"])
    terr_0 = info["terr_por_nivel"][0]
    terr_last = info["terr_por_nivel"][n_levels]

    # 1. Routing accuracy
    correct_n0 = sum(
        1
        for i, tid in enumerate(token_ids)
        if terr_0[i].argmax().item() == territory_table[tid].item()
    )
    correct_nlast = sum(
        1
        for i, tid in enumerate(token_ids)
        if terr_last[i].argmax().item() == territory_table[tid].item()
    )
    total = len(tokens)
    acc_n0 = correct_n0 / total * 100
    acc_nlast = correct_nlast / total * 100

    # 2. Routing margin
    margins = []
    for i in range(total):
        acts = terr_last[i].tolist()
        sorted_acts = sorted(acts, reverse=True)
        margins.append(sorted_acts[0] - sorted_acts[1])
    avg_margin = sum(margins) / len(margins)
    min_margin = min(margins)

    # 3. Routing std (diferenciación)
    stds = [terr_last[i].std().item() for i in range(total)]
    avg_std = sum(stds) / len(stds)

    # 4. Early Exit
    max_conf = max(info["confianza"])
    exit_ok = max_conf >= PRESET_V3.umbral_exit

    # 5. Stream balance
    dominant_counts = [0, 0, 0, 0]
    for i in range(total):
        d = terr_last[i].argmax().item()
        dominant_counts[d] += 1
    gini = _gini_coefficient(dominant_counts)

    def status(val: bool) -> str:
        return f"\033[32m● PASS{RESET}" if val else f"\033[31m● FAIL{RESET}"

    lines.append(f"  {'Métrica':<35} {'Valor':>10}  Estado")
    lines.append(f"  {'─' * 35} {'─' * 10}  {'─' * 12}")
    lines.append(
        f"  {'Routing accuracy N0':<35} {acc_n0:>9.1f}%  {status(acc_n0 >= 70)}"
    )
    lines.append(
        f"  {'Routing accuracy N' + str(n_levels):<35} {acc_nlast:>9.1f}%  {status(acc_nlast >= 70)}"
    )
    lines.append(
        f"  {'Margen promedio':<35} {avg_margin:>10.4f}  {status(avg_margin > 0.02)}"
    )
    lines.append(
        f"  {'Margen mínimo':<35} {min_margin:>10.4f}  {status(min_margin > 0.005)}"
    )
    lines.append(
        f"  {'Diferenciación (std promedio)':<35} {avg_std:>10.4f}  {status(avg_std > 0.02)}"
    )
    lines.append(
        f"  {'Early Exit max confianza':<35} {max_conf:>9.1%}  {status(exit_ok)}"
    )
    lines.append(
        f"  {'Diversidad routing (1-Gini)':<35} {1 - gini:>10.3f}  {status(gini < 0.6)}"
    )
    lines.append(
        f"  {'Distribución':<35} "
        + " ".join(
            f"{STREAM_COLORS[t]}{STREAM_NAMES[t][:4]}={dominant_counts[t]}{RESET}"
            for t in range(4)
        )
    )

    # Score global (0-100)
    score = (
        min(acc_nlast, 100) * 0.35
        + min(avg_margin * 1000, 100) * 0.20
        + min(avg_std * 1000, 100) * 0.15
        + (100 if exit_ok else max_conf * 100) * 0.15
        + (1 - gini) * 100 * 0.15
    )
    s_color = "\033[32m" if score >= 70 else "\033[33m" if score >= 40 else "\033[31m"
    lines.append(f"\n  {BOLD}Score global: {s_color}{score:.0f}/100{RESET}")

    return "\n".join(lines)


def _gini_coefficient(counts: list[int]) -> float:
    """Calcula coeficiente de Gini (0 = perfecto, 1 = todo en 1 clase)."""
    n = len(counts)
    total = sum(counts)
    if total == 0:
        return 0.0
    sorted_c = sorted(counts)
    cumulative = 0.0
    gini_sum = 0.0
    for c in sorted_c:
        cumulative += c
        gini_sum += cumulative
    return 1 - (2 * gini_sum - total) / (total * n)


# =============================================================================
# SUITE: BATERÍA DE TESTS DIVERSA
# =============================================================================


def ejecutar_suite(
    model: PamparV3,
    tokenizer: object,
    territory_table: torch.Tensor,
    device: torch.device,
) -> str:
    """Ejecuta la suite completa de código y agrega métricas."""
    lines = []
    lines.append(f"\n{BOLD}{'═' * 70}")
    lines.append(f"  SUITE DE DIAGNÓSTICO COMPLETA — {len(CODE_SUITE)} muestras")
    lines.append(f"{'═' * 70}{RESET}\n")

    all_correct_n0 = 0
    all_correct_nlast = 0
    all_total = 0
    all_margins: list[float] = []
    all_max_conf: list[float] = []
    per_sample: list[dict] = []

    for label, code in CODE_SUITE:
        token_ids = tokenizer.Encode(code, out_type=int)
        tokens_str = [tokenizer.IdToPiece(tid) for tid in token_ids]
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            info = forward_instrumentado(model, input_tensor)

        n_levels = len(info["confianza"])
        terr_0 = info["terr_por_nivel"][0]
        terr_last = info["terr_por_nivel"][n_levels]

        correct_n0 = 0
        correct_nlast = 0
        margins: list[float] = []

        for i, tid in enumerate(token_ids):
            expected = territory_table[tid].item()
            actual_n0 = terr_0[i].argmax().item()
            actual_nlast = terr_last[i].argmax().item()
            if actual_n0 == expected:
                correct_n0 += 1
            if actual_nlast == expected:
                correct_nlast += 1

            acts = terr_last[i].tolist()
            sorted_acts = sorted(acts, reverse=True)
            margins.append(sorted_acts[0] - sorted_acts[1])

        n = len(token_ids)
        acc_n0 = correct_n0 / n * 100
        acc_nlast = correct_nlast / n * 100
        avg_margin = sum(margins) / len(margins)
        max_conf = max(info["confianza"])

        all_correct_n0 += correct_n0
        all_correct_nlast += correct_nlast
        all_total += n
        all_margins.extend(margins)
        all_max_conf.append(max_conf)

        per_sample.append(
            {
                "label": label,
                "code": code.split("\n")[0][:40],
                "tokens": n,
                "acc_n0": acc_n0,
                "acc_nlast": acc_nlast,
                "margin": avg_margin,
                "max_conf": max_conf,
            }
        )

    # Tabla de resultados
    lines.append(
        f"  {'Muestra':<16} {'Código':<42} {'Tok':>3} {'AccN0':>6} {'AccN5':>6} {'Marg':>6} {'Exit':>5}"
    )
    lines.append(
        f"  {'─' * 16} {'─' * 42} {'─' * 3} {'─' * 6} {'─' * 6} {'─' * 6} {'─' * 5}"
    )

    for s in per_sample:
        c_n0 = "\033[32m" if s["acc_n0"] >= 70 else "\033[31m"
        c_nl = "\033[32m" if s["acc_nlast"] >= 70 else "\033[31m"
        c_m = "\033[32m" if s["margin"] > 0.02 else "\033[33m"
        c_e = "\033[32m" if s["max_conf"] >= 0.9 else "\033[31m"
        lines.append(
            f"  {s['label']:<16} {s['code']:<42} {s['tokens']:>3} "
            f"{c_n0}{s['acc_n0']:>5.1f}%{RESET} "
            f"{c_nl}{s['acc_nlast']:>5.1f}%{RESET} "
            f"{c_m}{s['margin']:>6.4f}{RESET} "
            f"{c_e}{s['max_conf']:>4.1%}{RESET}"
        )

    # Agregados
    global_acc_n0 = all_correct_n0 / all_total * 100
    global_acc_nlast = all_correct_nlast / all_total * 100
    global_margin = sum(all_margins) / len(all_margins)
    global_exit = sum(all_max_conf) / len(all_max_conf)
    min_acc = min(s["acc_nlast"] for s in per_sample)
    worst = [s for s in per_sample if s["acc_nlast"] == min_acc][0]

    lines.append(f"\n  {'─' * 90}")
    c_g = "\033[32m" if global_acc_nlast >= 70 else "\033[31m"
    lines.append(
        f"  {BOLD}GLOBAL{RESET}  Tokens: {all_total}  "
        f"AccN0: {global_acc_n0:.1f}%  "
        f"{BOLD}AccN5: {c_g}{global_acc_nlast:.1f}%{RESET}  "
        f"Margen: {global_margin:.4f}  "
        f"Exit promedio: {global_exit:.1%}"
    )
    lines.append(
        f"  {BOLD}Peor muestra:{RESET} {worst['label']} → acc={worst['acc_nlast']:.1f}%"
    )

    # Score
    score = (
        min(global_acc_nlast, 100) * 0.40
        + min(global_margin * 1000, 100) * 0.25
        + min(global_exit * 100, 100) * 0.15
        + min(min_acc, 100) * 0.20
    )
    s_color = "\033[32m" if score >= 70 else "\033[33m" if score >= 40 else "\033[31m"
    lines.append(f"\n  {BOLD}Score Suite: {s_color}{score:.0f}/100{RESET}")

    return "\n".join(lines)


# =============================================================================
# COMPARACIÓN DE CHECKPOINTS
# =============================================================================


def comparar_checkpoints(
    ckpt_a: Path,
    ckpt_b: Path,
    device: torch.device,
) -> str:
    """Compara dos checkpoints lado a lado con métricas clave."""
    lines = []
    lines.append(f"\n{BOLD}{'═' * 70}")
    lines.append(f"  COMPARACIÓN DE CHECKPOINTS")
    lines.append(f"{'═' * 70}{RESET}")
    lines.append(f"  A: {ckpt_a.name}")
    lines.append(f"  B: {ckpt_b.name}\n")

    results: dict[str, dict] = {}
    for label, path in [("A", ckpt_a), ("B", ckpt_b)]:
        model, tokenizer = load_model(path, device, verbose=False)
        territory_table = _build_territory_table(tokenizer)

        acc_n0_total = 0
        acc_nlast_total = 0
        total_tokens = 0
        margins: list[float] = []
        confs: list[float] = []

        for _, code in CODE_SUITE:
            token_ids = tokenizer.Encode(code, out_type=int)
            input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                info = forward_instrumentado(model, input_tensor)

            n_levels = len(info["confianza"])
            terr_0 = info["terr_por_nivel"][0]
            terr_last = info["terr_por_nivel"][n_levels]

            for i, tid in enumerate(token_ids):
                expected = territory_table[tid].item()
                if terr_0[i].argmax().item() == expected:
                    acc_n0_total += 1
                if terr_last[i].argmax().item() == expected:
                    acc_nlast_total += 1
                acts = terr_last[i].tolist()
                sorted_a = sorted(acts, reverse=True)
                margins.append(sorted_a[0] - sorted_a[1])
                total_tokens += 1

            confs.append(max(info["confianza"]))

        # Diferenciación promedio (std de routing)
        avg_std = 0.0
        std_count = 0
        for _, code in CODE_SUITE[:5]:
            token_ids = tokenizer.Encode(code, out_type=int)
            input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                info2 = forward_instrumentado(model, input_tensor)
            n_levels = len(info2["confianza"])
            for i in range(len(token_ids)):
                avg_std += info2["terr_por_nivel"][n_levels][i].std().item()
                std_count += 1
        avg_std /= max(std_count, 1)

        results[label] = {
            "acc_n0": acc_n0_total / total_tokens * 100,
            "acc_nlast": acc_nlast_total / total_tokens * 100,
            "margin": sum(margins) / len(margins),
            "min_margin": min(margins),
            "exit_avg": sum(confs) / len(confs),
            "exit_max": max(confs),
            "diff_std": avg_std,
        }

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Tabla comparativa
    lines.append(
        f"  {'Métrica':<30} {'Ckpt A':>10} {'Ckpt B':>10} {'Delta':>10}  Mejor"
    )
    lines.append(f"  {'─' * 30} {'─' * 10} {'─' * 10} {'─' * 10}  {'─' * 6}")

    metrics = [
        ("Accuracy N0", "acc_n0", "%", True),
        ("Accuracy N5", "acc_nlast", "%", True),
        ("Margen promedio", "margin", "", True),
        ("Margen mínimo", "min_margin", "", True),
        ("Exit promedio", "exit_avg", "%", True),
        ("Exit máximo", "exit_max", "%", True),
        ("Diferenciación (std)", "diff_std", "", True),
    ]

    for name, key, unit, higher_better in metrics:
        va = results["A"][key]
        vb = results["B"][key]
        delta = vb - va
        is_pct = unit == "%"
        fmt = ".1f" if is_pct else ".4f"
        suf = "%" if is_pct else ""

        better = "B" if (delta > 0) == higher_better else "A" if delta != 0 else "="
        b_color = "\033[32m" if better == "B" else "\033[33m" if better == "A" else ""
        d_sign = "+" if delta > 0 else ""

        lines.append(
            f"  {name:<30} {va:>9{fmt}}{suf} {vb:>9{fmt}}{suf} "
            f"{b_color}{d_sign}{delta:>9{fmt}}{suf}{RESET}  {better}"
        )

    return "\n".join(lines)


# =============================================================================
# GENERACIÓN DE CÓDIGO
# =============================================================================


def mostrar_generacion(
    model: PamparV3,
    tokenizer: object,
    prompt: str,
    device: torch.device,
    max_tokens: int = 128,
    temperature: float = 0.7,
) -> str:
    """Genera código desde un prompt y muestra el resultado."""
    lines = []
    lines.append(f"\n{BOLD}═══ GENERACIÓN DE CÓDIGO ═══{RESET}")
    lines.append(f"  Prompt: {prompt}")
    lines.append(f"  Params: max_tokens={max_tokens}, temperature={temperature}\n")

    prompt_ids = tokenizer.Encode(prompt, out_type=int)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    generated_ids = output_ids[0].tolist()
    generated_text = tokenizer.Decode(generated_ids)

    # Separar prompt del generado
    prompt_text = tokenizer.Decode(prompt_ids)
    new_text = generated_text[len(prompt_text) :]

    lines.append(f"  {DIM}{'─' * 60}{RESET}")
    lines.append(f"  {DIM}{prompt_text}{RESET}{BOLD}{new_text}{RESET}")
    lines.append(f"  {DIM}{'─' * 60}{RESET}")
    lines.append(f"  Tokens generados: {len(generated_ids) - len(prompt_ids)}")

    # Análisis del routing de lo generado
    with torch.no_grad():
        info = forward_instrumentado(
            model, output_ids[:, : min(64, output_ids.shape[1])]
        )

    n_levels = len(info["confianza"])
    terr_last = info["terr_por_nivel"][n_levels]
    gen_start = len(prompt_ids)
    gen_end = min(64, output_ids.shape[1])

    if gen_end > gen_start:
        dominant_counts = [0, 0, 0, 0]
        for i in range(gen_start, gen_end):
            d = terr_last[i].argmax().item()
            dominant_counts[d] += 1
        n_gen = gen_end - gen_start
        lines.append(
            f"\n  Routing generado: "
            + " ".join(
                f"{STREAM_COLORS[t]}{STREAM_NAMES[t][:4]}={dominant_counts[t]}/{n_gen}{RESET}"
                for t in range(4)
            )
        )

    return "\n".join(lines)


# =============================================================================
# VISUALIZACIÓN: PESOS DEL MODELO
# =============================================================================


def mostrar_pesos(model: PamparV3) -> str:
    """Muestra distribución de pesos por componente del modelo."""
    lines = []
    lines.append(f"\n{BOLD}═══ ANATOMÍA DE PESOS ═══{RESET}\n")

    stats = model.count_params()
    total = stats["total"]

    componentes = {
        "Embedding (tok_emb)": stats["embeddings"],
        "Tálamo Inicial": stats["talamo_inicial"],
        "Niveles (5×NivelProfundo)": stats["niveles"],
        "Norm Final": stats["norm_f"],
    }

    lines.append(f"  {'Componente':<30} {'Params':>12} {'%':>8}  {'Distribución':>20}")
    lines.append(f"  {'─' * 30} {'─' * 12} {'─' * 8}  {'─' * 20}")

    for name, count in componentes.items():
        pct = count / total
        lines.append(f"  {name:<30} {count:>12,} {pct:>7.1%}  {barra(pct, 20)}")

    lines.append(f"\n  {BOLD}Total: {total:,} parámetros ({total / 1e6:.1f}M){RESET}")

    # Detalle por nivel
    lines.append(f"\n  {BOLD}Detalle por nivel:{RESET}")
    for i, nivel in enumerate(model.niveles):
        attn_p = sum(p.numel() for p in nivel.attn.parameters())
        ffn_p = sum(p.numel() for p in nivel.ffns.parameters())
        lat_p = sum(p.numel() for p in nivel.lateral.parameters())
        tal_p = sum(p.numel() for p in nivel.talamo_nivel.parameters())
        exit_p = sum(p.numel() for p in nivel.exit_head.parameters())
        nivel_total = attn_p + ffn_p + lat_p + tal_p + exit_p

        lines.append(
            f"\n  Nivel {i}:  {nivel_total:,} params ({nivel_total / 1e6:.1f}M)"
        )
        lines.append(
            f"    Atención GQA:   {attn_p:>10,}  {barra(attn_p / nivel_total, 15)}"
        )
        lines.append(
            f"    4× StreamFFN:   {ffn_p:>10,}  {barra(ffn_p / nivel_total, 15)}"
        )
        lines.append(
            f"    LateralGate:    {lat_p:>10,}  {barra(lat_p / nivel_total, 15)}"
        )
        lines.append(
            f"    TalamoNivel:    {tal_p:>10,}  {barra(tal_p / nivel_total, 15)}"
        )
        lines.append(
            f"    Exit Head:      {exit_p:>10,}  {barra(exit_p / nivel_total, 15)}"
        )

    # Distribución de magnitud de pesos
    lines.append(f"\n  {BOLD}Salud de pesos (magnitud):{RESET}")
    for name, param in model.named_parameters():
        if param.numel() < 1000:
            continue
        data = param.detach().float().cpu()
        mean_abs = data.abs().mean().item()
        std = data.std().item()
        dead = (data.abs() < 1e-6).float().mean().item()

        # Alertas
        alert = ""
        if dead > 0.5:
            alert = f" \033[31m⚠ {dead:.0%} muertos{RESET}"
        elif std < 1e-5:
            alert = f" \033[33m⚠ baja varianza{RESET}"

        if alert:
            short_name = (
                name.replace("niveles.", "N")
                .replace("ffns.", "FFN")
                .replace("lateral.", "Lat.")
            )
            lines.append(
                f"    {short_name:<45} μ|w|={mean_abs:.4f}  σ={std:.4f}{alert}"
            )

    return "\n".join(lines)


# =============================================================================
# EXPORTAR HTML
# =============================================================================


def exportar_html(
    tokens: list[str],
    token_ids: list[int],
    info: dict,
    output_path: Path,
    code: str,
    territory_table: torch.Tensor,
) -> None:
    """Exporta el scan como un HTML auto-contenido con métricas de precisión."""
    n_levels = len(info["confianza"])

    # Heatmap territorial por token (nivel 0)
    terr_0 = info["terr_por_nivel"][0]
    terr_last = info["terr_por_nivel"][n_levels]

    rows_html = []
    correct_n0 = 0
    correct_nlast = 0
    total = len(tokens)
    margins = []

    for i, (tok, tid) in enumerate(zip(tokens, token_ids)):
        acts = terr_0[i].tolist()
        acts_last = terr_last[i].tolist()
        dominant = max(range(4), key=lambda t: acts[t])
        dominant_last = max(range(4), key=lambda t: acts_last[t])
        expected = territory_table[tid].item()

        match_n0 = dominant == expected
        match_nlast = dominant_last == expected
        if match_n0:
            correct_n0 += 1
        if match_nlast:
            correct_nlast += 1

        sorted_a = sorted(acts_last, reverse=True)
        margins.append(sorted_a[0] - sorted_a[1])

        cells = "".join(
            f'<td style="background:rgba({_stream_rgb(t)},{acts[t]:.2f})">{acts[t]:.3f}</td>'
            for t in range(4)
        )
        match_sym = "✓" if match_nlast else "✗"
        match_color = "#a6e3a1" if match_nlast else "#f38ba8"
        tok_esc = tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        rows_html.append(
            f'<tr><td class="tok">{tok_esc}</td>{cells}'
            f'<td class="dom" style="color:{_stream_hex(dominant)}">{STREAM_NAMES[dominant]}</td>'
            f'<td style="color:{_stream_hex(expected)}">{STREAM_NAMES[expected]}</td>'
            f'<td style="color:{match_color};font-weight:bold">{match_sym}</td></tr>'
        )

    acc_n0 = correct_n0 / total * 100 if total > 0 else 0
    acc_nlast = correct_nlast / total * 100 if total > 0 else 0
    avg_margin = sum(margins) / len(margins) if margins else 0

    # Evolución heatmap
    evo_rows = []
    for i, tok in enumerate(tokens):
        tok_esc = tok.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        cells = ""
        for n in range(n_levels + 1):
            acts = info["terr_por_nivel"][n][i].tolist()
            dominant = max(range(4), key=lambda t: acts[t])
            cells += f'<td style="background:{_stream_hex(dominant)};opacity:{max(acts):.2f}">{max(acts):.2f}</td>'
        evo_rows.append(f'<tr><td class="tok">{tok_esc}</td>{cells}</tr>')

    # Confianza
    conf_bars = ""
    max_conf = 0.0
    for n, conf in enumerate(info["confianza"]):
        color = "#4caf50" if conf >= PRESET_V3.umbral_exit else "#f44336"
        conf_bars += f'<div class="conf-bar"><span>Nivel {n}</span><div class="bar" style="width:{conf * 100:.1f}%;background:{color}"></div><span>{conf:.1%}</span></div>'
        max_conf = max(max_conf, conf)

    # Score
    stds = [terr_last[i].std().item() for i in range(total)]
    avg_std = sum(stds) / len(stds) if stds else 0
    score = (
        min(acc_nlast, 100) * 0.35
        + min(avg_margin * 1000, 100) * 0.20
        + min(avg_std * 1000, 100) * 0.15
        + (100 if max_conf >= 0.9 else max_conf * 100) * 0.15
        + 50 * 0.15
    )
    score_color = "#a6e3a1" if score >= 70 else "#f9e2af" if score >= 40 else "#f38ba8"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>PAMPAr Brain Scanner</title>
<style>
body {{ font-family: 'Cascadia Code', 'Fira Code', monospace; background: #1e1e2e; color: #cdd6f4; margin: 2em; }}
h1 {{ color: #89b4fa; border-bottom: 2px solid #89b4fa; padding-bottom: 8px; }}
h2 {{ color: #a6e3a1; margin-top: 2em; }}
.code {{ background: #313244; padding: 1em; border-radius: 8px; font-size: 14px; white-space: pre; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
td, th {{ padding: 4px 8px; border: 1px solid #45475a; font-size: 13px; }}
th {{ background: #313244; }}
.tok {{ background: #313244; font-weight: bold; white-space: pre; }}
.dom {{ font-weight: bold; }}
.conf-bar {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
.conf-bar .bar {{ height: 20px; border-radius: 4px; transition: width 0.3s; }}
.conf-bar span {{ min-width: 60px; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1em; margin: 1em 0; }}
.metric {{ background: #313244; padding: 1em; border-radius: 8px; text-align: center; }}
.metric .value {{ font-size: 2em; font-weight: bold; }}
.metric .label {{ font-size: 0.85em; color: #a6adc8; margin-top: 4px; }}
</style>
</head>
<body>
<h1>🧠 PAMPAr Brain Scanner</h1>
<div class="code">{code.replace("&", "&amp;").replace("<", "&lt;")}</div>

<h2>Métricas de Salud</h2>
<div class="metrics">
  <div class="metric"><div class="value" style="color:{score_color}">{score:.0f}/100</div><div class="label">Score Global</div></div>
  <div class="metric"><div class="value" style="color:{"#a6e3a1" if acc_nlast >= 70 else "#f38ba8"}">{acc_nlast:.1f}%</div><div class="label">Accuracy Routing</div></div>
  <div class="metric"><div class="value" style="color:{"#a6e3a1" if avg_margin > 0.02 else "#f9e2af"}">{avg_margin:.4f}</div><div class="label">Margen Promedio</div></div>
  <div class="metric"><div class="value" style="color:{"#a6e3a1" if max_conf >= 0.9 else "#f38ba8"}">{max_conf:.1%}</div><div class="label">Early Exit Max</div></div>
  <div class="metric"><div class="value">{avg_std:.4f}</div><div class="label">Diferenciación</div></div>
</div>

<h2>Tálamo: Routing Inicial</h2>
<table>
<tr><th>Token</th><th>SINTAXIS</th><th>SEMANTICA</th><th>LOGICO</th><th>ESTRUCTURAL</th><th>Actual</th><th>Esperado</th><th>OK</th></tr>
{"".join(rows_html)}
</table>

<h2>Evolución por Nivel</h2>
<table>
<tr><th>Token</th>{"".join(f"<th>N{n}</th>" for n in range(n_levels + 1))}</tr>
{"".join(evo_rows)}
</table>

<h2>Early Exit</h2>
{conf_bars}
<p>Umbral: {PRESET_V3.umbral_exit:.0%} — Mín {PRESET_V3.capas_min} niveles</p>

</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")


def _stream_rgb(idx: int) -> str:
    """RGB para cada stream (sin alpha)."""
    return ["137,180,250", "166,227,161", "249,226,175", "203,166,247"][idx]


def _stream_hex(idx: int) -> str:
    """Color hex para cada stream."""
    return ["#89b4fa", "#a6e3a1", "#f9e2af", "#cba6f7"][idx]


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PAMPAr Brain Scanner — Diagnóstico completo de la arquitectura cerebral",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--code",
        type=str,
        default=None,
        help="Código Python a analizar (ej: 'def fibonacci(n):')",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Ejecutar suite completa de diagnóstico (20 muestras diversas)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("CKPT_A", "CKPT_B"),
        help="Comparar dos checkpoints lado a lado",
    )
    parser.add_argument(
        "--generate",
        type=str,
        default=None,
        help="Generar código desde un prompt y analizar routing",
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        help="Mostrar distribución de pesos del modelo",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "v3_sft_v8.pt"),
        help="Path al checkpoint (.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--html",
        type=str,
        default=None,
        help="Exportar resultado como HTML (ej: scan.html)",
    )

    args = parser.parse_args()

    has_action = (
        args.code or args.weights or args.suite or args.compare or args.generate
    )
    if not has_action:
        parser.error(
            "Necesitas al menos uno de: --code, --suite, --compare, --generate, --weights"
        )

    # Resolver device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{BOLD}🧠 PAMPAr Brain Scanner v2{RESET}")
    print(f"   Device: {device}")

    # --- Comparación de checkpoints (no necesita cargar modelo) ---
    if args.compare:
        ckpt_a = Path(args.compare[0])
        ckpt_b = Path(args.compare[1])
        for p in [ckpt_a, ckpt_b]:
            if not p.exists():
                print(f"\033[31mError: Checkpoint no encontrado: {p}{RESET}")
                sys.exit(1)
        print(comparar_checkpoints(ckpt_a, ckpt_b, device))
        print()
        return

    # Cargar modelo para los demás modos
    print(f"   Checkpoint: {args.checkpoint}\n")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"\033[31mError: Checkpoint no encontrado: {ckpt_path}{RESET}")
        sys.exit(1)

    model, tokenizer = load_model(ckpt_path, device, verbose=False)
    print(
        f"   Modelo cargado: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
    )

    # Construir territory table
    territory_table = _build_territory_table(tokenizer)
    print(f"   Territory table: {territory_table.shape[0]} tokens mapeados\n")

    # --- Análisis de pesos ---
    if args.weights:
        print(mostrar_pesos(model))

    # --- Suite de diagnóstico ---
    if args.suite:
        print(ejecutar_suite(model, tokenizer, territory_table, device))

    # --- Generación ---
    if args.generate:
        print(mostrar_generacion(model, tokenizer, args.generate, device))

    # --- Análisis de código ---
    if args.code:
        tokens_ids = tokenizer.Encode(args.code, out_type=int)
        tokens_str = [tokenizer.IdToPiece(tid) for tid in tokens_ids]

        print(f"   Código: {BOLD}{args.code}{RESET}")
        print(f"   Tokens: {len(tokens_str)} → {tokens_str}\n")

        input_tensor = torch.tensor([tokens_ids], dtype=torch.long, device=device)
        info = forward_instrumentado(model, input_tensor)

        # Mostrar todas las visualizaciones
        print(mostrar_routing(tokens_str, info))
        print(mostrar_precision(tokens_str, tokens_ids, info, territory_table))
        print(mostrar_margen(tokens_str, info))
        print(mostrar_zonas(tokens_str, info))
        print(mostrar_evolucion(tokens_str, info))
        print(mostrar_fibras_blancas(info))
        print(mostrar_early_exit(info))
        print(mostrar_stream_norms(info))
        print(mostrar_resumen(tokens_str, tokens_ids, info, territory_table))

        # Exportar HTML si se pidió
        if args.html:
            html_path = Path(args.html)
            exportar_html(
                tokens_str, tokens_ids, info, html_path, args.code, territory_table
            )
            print(f"\n   {BOLD}HTML exportado:{RESET} {html_path.resolve()}")

    print()


if __name__ == "__main__":
    main()
