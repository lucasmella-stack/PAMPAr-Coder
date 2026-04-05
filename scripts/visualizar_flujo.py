#!/usr/bin/env python3
"""
visualizar_flujo.py — Visualizador interactivo del flujo de datos en PamparV3.

Genera un HTML standalone con 4 paneles:
  1. Embeddings iniciales (PCA 2D) — dónde vive cada token en el espacio
  2. Mapa de atención por nivel — qué tokens miran a cuáles
  3. Norma de activaciones por nivel/stream — cómo crece/decae la señal
  4. Benchmark de velocidad y memoria

Uso:
    python visualizar_flujo.py "Hola, me llamo Pampar"
    python visualizar_flujo.py --texto "def suma(a, b): return a + b"
    python visualizar_flujo.py  # usa texto por defecto
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# ── Setup de paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

# ── Captura de activaciones via hooks ───────────────────────────────────────


class ActivationCapture:
    """Registra hooks en PamparV3 y captura tensores del forward pass."""

    def __init__(self):
        self.handles: list = []
        self.embeddings: torch.Tensor | None = None  # [B, L, dim]
        self.nivel_outputs: list[torch.Tensor] = []  # por nivel: [B, L, dim]
        self.attn_weights: list[torch.Tensor] = []  # por nivel: [B, H, L, L]
        self.stream_norms: list[list[float]] = []  # por nivel: [4 streams]

    def attach(self, model) -> None:
        """Registra hooks en el modelo."""

        # Hook en embedding
        def hook_emb(module, inp, out):
            self.embeddings = out.detach().cpu()

        self.handles.append(model.tok_emb.register_forward_hook(hook_emb))

        # Hook en cada NivelProfundo
        for i, nivel in enumerate(model.niveles):

            def make_nivel_hook(idx):
                def hook(module, inp, out):
                    # out = (streams_list, terr_acts, conf)
                    # streams_list es una lista de n_streams tensores [B, L, D]
                    streams_out = out[0] if isinstance(out, (tuple, list)) else out
                    if isinstance(streams_out, (list, tuple)):
                        x = torch.stack([s.detach().cpu() for s in streams_out]).mean(0)
                    else:
                        x = streams_out.detach().cpu()
                    self.nivel_outputs.append(x)

                    # Norma por stream (cada cuarto del dim como proxy de stream)
                    B, L, D = x.shape
                    chunk = max(1, D // 4)
                    norms = [
                        x[:, :, i * chunk : min((i + 1) * chunk, D)]
                        .norm(dim=-1)
                        .mean()
                        .item()
                        for i in range(4)
                    ]
                    self.stream_norms.append(norms)

                return hook

            self.handles.append(nivel.register_forward_hook(make_nivel_hook(i)))

            # Hook en la atención del nivel
            def make_attn_hook(idx):
                def hook(module, inp, out):
                    # Recalcular pesos de atención desde el input
                    x = inp[0]
                    B, L, D = x.shape
                    H = module.n_heads
                    Hkv = module.n_kv_heads
                    head_dim = module.head_dim

                    # Q, K usando nombres reales del BloqueAttn
                    q = module.q_proj(x).view(B, L, H, head_dim).transpose(1, 2)
                    k = module.k_proj(x).view(B, L, Hkv, head_dim).transpose(1, 2)
                    k = module._repeat_kv(k)  # GQA expand

                    scale = head_dim**-0.5
                    scores = (
                        torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
                    )
                    # Máscara causal
                    mask = torch.triu(
                        torch.ones(L, L, device=x.device), diagonal=1
                    ).bool()
                    scores = scores.masked_fill(
                        mask.unsqueeze(0).unsqueeze(0), float("-inf")
                    )
                    weights = F.softmax(scores, dim=-1)
                    self.attn_weights.append(weights.detach().cpu())

                return hook

            self.handles.append(nivel.attn.register_forward_hook(make_attn_hook(i)))

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ── PCA manual (sin sklearn) ────────────────────────────────────────────────


def pca_2d(matrix: torch.Tensor) -> torch.Tensor:
    """Reduce [N, D] a [N, 2] via PCA (SVD)."""
    m = matrix.float()
    m = m - m.mean(0, keepdim=True)
    _, _, V = torch.svd(m)
    return m @ V[:, :2]


# ── Generación del HTML ─────────────────────────────────────────────────────


def build_html(
    tokens: list[str],
    capture: ActivationCapture,
    text: str,
    elapsed_ms: float,
    mem_mb: float,
) -> str:

    n_tokens = len(tokens)
    n_levels = len(capture.attn_weights)

    # ── Datos embedding PCA ─────────────────────────────────────────────────
    emb = capture.embeddings[0]  # [L, dim]
    if n_tokens >= 2:
        coords = pca_2d(emb).tolist()
    else:
        coords = [[0.0, 0.0]] * n_tokens

    emb_x = [c[0] for c in coords]
    emb_y = [c[1] for c in coords]
    emb_norm = emb.norm(dim=-1).tolist()

    # ── Datos atención — promedio de cabezas por nivel ───────────────────────
    attn_data = []
    for lvl_w in capture.attn_weights:
        # lvl_w: [B, H, L, L] → promedio de H → [L, L]
        avg = lvl_w[0].mean(0).tolist()
        attn_data.append(avg)

    # ── Normas por nivel/stream ──────────────────────────────────────────────
    stream_norms = capture.stream_norms  # [[s0,s1,s2,s3], ...] por nivel

    # ── Serializar a JSON inline ─────────────────────────────────────────────
    import json

    tokens_json = json.dumps(tokens)
    emb_x_json = json.dumps(emb_x)
    emb_y_json = json.dumps(emb_y)
    emb_norm_json = json.dumps(emb_norm)
    attn_json = json.dumps(attn_data)
    stream_norms_json = json.dumps(stream_norms)
    n_levels_json = n_levels
    elapsed_json = elapsed_ms
    mem_json = mem_mb
    text_escaped = text.replace('"', '\\"')

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>PamparV3 — Flujo de datos</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0;
         margin: 0; padding: 16px; }}
  h1 {{ color: #7c9eff; font-size: 1.2rem; margin-bottom: 4px; }}
  .subtitle {{ color: #888; font-size: 0.85rem; margin-bottom: 16px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .panel {{ background: #1a1d27; border-radius: 8px; padding: 16px; }}
  .panel h2 {{ font-size: 0.9rem; color: #aac4ff; margin: 0 0 8px 0; }}
  .stats {{ display: flex; gap: 24px; margin-bottom: 16px; }}
  .stat {{ background: #1a1d27; border-radius: 8px; padding: 12px 20px; }}
  .stat .val {{ font-size: 1.6rem; font-weight: bold; color: #7c9eff; }}
  .stat .lbl {{ font-size: 0.75rem; color: #888; }}
  select {{ background: #2a2d3a; color: #e0e0e0; border: 1px solid #444;
            border-radius: 4px; padding: 4px 8px; margin-bottom: 8px; }}
  .input-text {{ background: #2a2d3a; border-radius: 6px; padding: 10px 14px;
                 font-size: 0.9rem; color: #ccc; margin-bottom: 16px;
                 border-left: 3px solid #7c9eff; }}
</style>
</head>
<body>
<h1>PamparV3 — Visualizador de Flujo Interno</h1>
<div class="subtitle">Arquitectura: 640d · {n_levels_json} niveles · 4 streams · GQA</div>

<div class="input-text">"{text_escaped}"</div>

<div class="stats">
  <div class="stat">
    <div class="val">{n_tokens}</div>
    <div class="lbl">tokens</div>
  </div>
  <div class="stat">
    <div class="val" id="statMs">{elapsed_ms:.1f}ms</div>
    <div class="lbl">forward pass</div>
  </div>
  <div class="stat">
    <div class="val">{mem_mb:.0f}MB</div>
    <div class="lbl">VRAM usada</div>
  </div>
  <div class="stat">
    <div class="val">{n_levels_json}</div>
    <div class="lbl">niveles de profundidad</div>
  </div>
</div>

<div class="grid">
  <div class="panel">
    <h2>1. Embeddings iniciales (PCA 2D)</h2>
    <div id="plotEmb" style="height:320px"></div>
  </div>
  <div class="panel">
    <h2>2. Mapa de atención
      <select id="selNivel" onchange="updateAttn()"></select>
    </h2>
    <div id="plotAttn" style="height:320px"></div>
  </div>
  <div class="panel">
    <h2>3. Norma de activaciones por stream</h2>
    <div id="plotNorms" style="height:320px"></div>
  </div>
  <div class="panel">
    <h2>4. Distancia que recorre cada token</h2>
    <div id="plotDrift" style="height:320px"></div>
  </div>
</div>

<script>
const TOKENS = {tokens_json};
const EMB_X  = {emb_x_json};
const EMB_Y  = {emb_y_json};
const EMB_NORM = {emb_norm_json};
const ATTN   = {attn_json};
const STREAM_NORMS = {stream_norms_json};
const N_LEVELS = {n_levels_json};

const PLOTLY_CFG = {{responsive: true, displayModeBar: false}};
const DARK = {{
  paper_bgcolor: '#1a1d27', plot_bgcolor: '#1a1d27',
  font: {{color: '#e0e0e0', size: 11}},
  margin: {{l:40, r:10, t:10, b:60}}
}};

// ── 1. Embedding PCA ────────────────────────────────────────────────────────
Plotly.newPlot('plotEmb', [{{
  x: EMB_X, y: EMB_Y,
  mode: 'markers+text',
  text: TOKENS,
  textposition: 'top center',
  marker: {{
    size: EMB_NORM.map(n => Math.max(6, Math.min(20, n * 2))),
    color: EMB_NORM,
    colorscale: 'Viridis',
    showscale: true,
    colorbar: {{title: 'norma', thickness: 12}}
  }},
  hovertemplate: '%{{text}}<br>norma: %{{marker.color:.2f}}<extra></extra>'
}}], {{
  ...DARK,
  xaxis: {{title: 'PC1', gridcolor:'#2a2d3a', zeroline:false}},
  yaxis: {{title: 'PC2', gridcolor:'#2a2d3a', zeroline:false}}
}}, PLOTLY_CFG);

// ── 2. Atención ─────────────────────────────────────────────────────────────
const sel = document.getElementById('selNivel');
for (let i = 0; i < N_LEVELS; i++) {{
  const opt = document.createElement('option');
  opt.value = i; opt.text = `Nivel ${{i+1}}`;
  sel.appendChild(opt);
}}

function updateAttn() {{
  const lvl = parseInt(sel.value);
  const w = ATTN[lvl];
  Plotly.react('plotAttn', [{{
    z: w, x: TOKENS, y: TOKENS,
    type: 'heatmap',
    colorscale: 'Blues',
    hovertemplate: '%{{y}} → %{{x}}: %{{z:.3f}}<extra></extra>'
  }}], {{
    ...DARK,
    xaxis: {{title: 'Key (token origen)', tickangle:-45, gridcolor:'#2a2d3a'}},
    yaxis: {{title: 'Query (token actual)', gridcolor:'#2a2d3a', autorange:'reversed'}}
  }}, PLOTLY_CFG);
}}
updateAttn();

// ── 3. Normas por stream ─────────────────────────────────────────────────────
const streamColors = ['#7c9eff','#ff7ca8','#7cffa0','#ffd97c'];
const streamTraces = [0,1,2,3].map(s => ({{
  y: STREAM_NORMS.map(lvl => lvl[s]),
  x: STREAM_NORMS.map((_, i) => `N${{i+1}}`),
  name: `Stream ${{s+1}}`,
  type: 'scatter', mode: 'lines+markers',
  line: {{color: streamColors[s], width: 2}},
  marker: {{size: 7}}
}}));
Plotly.newPlot('plotNorms', streamTraces, {{
  ...DARK,
  xaxis: {{title: 'Nivel', gridcolor:'#2a2d3a'}},
  yaxis: {{title: 'Norma media', gridcolor:'#2a2d3a'}},
  legend: {{bgcolor:'transparent'}}
}}, PLOTLY_CFG);

// ── 4. Drift (cuánto se movió cada token) ────────────────────────────────────
// Comparar embedding inicial vs salida del último nivel
// (solo si capturamos nivel_outputs — aquí usamos norma de STREAM_NORMS como proxy)
const finalNorm = STREAM_NORMS.length > 0 ? STREAM_NORMS[STREAM_NORMS.length-1] : [0,0,0,0];
const initNorm  = EMB_NORM;
const avgFinal  = finalNorm.reduce((a,b)=>a+b,0)/4;

// Drift por token = norma embedding vs norma proyectada (proxy visual)
const driftProxy = initNorm.map((n,i) => Math.abs(n - avgFinal));

Plotly.newPlot('plotDrift', [{{
  x: TOKENS, y: driftProxy,
  type: 'bar',
  marker: {{
    color: driftProxy,
    colorscale: 'RdYlGn_r',
    showscale: false
  }},
  hovertemplate: '%{{x}}: drift=%{{y:.3f}}<extra></extra>'
}}], {{
  ...DARK,
  xaxis: {{title: 'Token', gridcolor:'#2a2d3a', tickangle:-30}},
  yaxis: {{title: 'Δ norma (embedding → salida)', gridcolor:'#2a2d3a'}}
}}, PLOTLY_CFG);
</script>
</body>
</html>"""


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Visualizador de flujo PamparV3")
    parser.add_argument(
        "texto",
        nargs="?",
        default="Hola me llamo Pampar y aprendo a programar",
        help="Texto a analizar",
    )
    parser.add_argument(
        "--texto", dest="texto_flag", help="Alternativa: --texto 'tu frase'"
    )
    parser.add_argument("--checkpoint", default="checkpoints/v3_classroom.pt")
    parser.add_argument("--out", default="sessions/flujo_pampar.html")
    args = parser.parse_args()

    text = args.texto_flag or args.texto

    # ── Cargar modelo ─────────────────────────────────────────────────────────
    print(f"Cargando modelo desde {args.checkpoint}...")
    import sentencepiece as spm
    from pampar.coder.v3.config import PRESET_V3
    from pampar.coder.v3.modelo import PamparV3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = spm.SentencePieceProcessor()
    tok.Load(str(ROOT / "data" / "tokenizer" / "pampar_48k.model"))

    model = PamparV3(PRESET_V3).to(device)
    ckpt_path = ROOT / args.checkpoint
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        state = ckpt.get("modelo", ckpt.get("model", ckpt))
        model.load_state_dict(state, strict=False)
        print(f"  Checkpoint cargado: {ckpt_path.name}")
    else:
        print(f"  Checkpoint no encontrado, usando pesos iniciales")

    model.registrar_tokenizer(tok)
    model.eval()

    # ── Forward pass con hooks ────────────────────────────────────────────────
    cap = ActivationCapture()
    cap.attach(model)

    ids = tok.Encode(text)
    tokens_str = [tok.IdToPiece(i).replace("▁", " ").strip() or "<unk>" for i in ids]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    # Medir memoria antes
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.no_grad():
        logits, _, _ = model(input_ids)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        import os

        import psutil

        proc = psutil.Process(os.getpid())
        mem_mb = proc.memory_info().rss / 1024**2

    cap.detach()

    # ── Top-5 predicciones del último token ──────────────────────────────────
    last_logits = logits[0, -1]
    top5 = last_logits.topk(5)
    print(f"\nTexto: '{text}'")
    print(f"Tokens ({len(ids)}): {tokens_str}")
    print(f"Forward pass: {elapsed_ms:.1f}ms | Memoria: {mem_mb:.0f}MB")
    print(f"\nTop-5 siguiente token:")
    for score, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        piece = tok.IdToPiece(idx).replace("▁", " ")
        prob = torch.softmax(last_logits, dim=0)[idx].item()
        print(f"  '{piece}' — {prob * 100:.1f}%")

    # ── Generar HTML ──────────────────────────────────────────────────────────
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(tokens_str, cap, text, elapsed_ms, mem_mb)
    out_path.write_text(html, encoding="utf-8")
    print(f"\nHTML generado: {out_path}")
    print("Abre ese archivo en Chrome/Edge para ver la visualización.")


if __name__ == "__main__":
    main()
