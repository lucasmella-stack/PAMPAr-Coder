# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
GhidraProbe — Instrumentación read-only del forward pass de PamparV3.

Registra forward hooks en cada componente clave del modelo para capturar
activaciones sin modificar el forward pass original. Cero parámetros extra.

Captura:
  - TalamoInicial: llaves_acts, attn_acts, zona_acts, terr_acts
  - NivelProfundo × 5: streams in/out, FFN deltas, lateral deltas, exit conf
  - StreamFFN × 4 × 5: delta por stream
  - LateralGate × 5: contribución lateral y scale learneable

Uso:
    probe = GhidraProbe(model)
    logits, loss, info = model(input_ids)
    report = probe.report()    # dict con todo
    probe.print_summary()      # resumen legible
    probe.detach()              # limpiar hooks
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .modelo import PamparV3

STREAM_NAMES = ("SINT", "SEMA", "LOGI", "ESTR")


@dataclass
class NivelCapture:
    """Capturas de un NivelProfundo."""

    nivel_idx: int = 0
    # Streams antes y después del nivel
    streams_in_norms: List[float] = field(default_factory=list)   # [4] L2 norm promedio
    streams_out_norms: List[float] = field(default_factory=list)  # [4]
    ffn_delta_norms: List[float] = field(default_factory=list)    # [4] cuánto cambió cada FFN
    lateral_delta_norms: List[float] = field(default_factory=list)  # [4]
    lateral_scales: List[float] = field(default_factory=list)     # [4] scale learnable actual
    # Routing
    terr_acts_mean: List[float] = field(default_factory=list)     # [4] activación media por territorio
    terr_acts_max_stream: int = 0       # territorio dominante
    # Exit
    exit_conf_mean: float = 0.0
    exit_conf_min: float = 0.0
    exit_conf_p10: float = 0.0          # percentil 10 (lo que usa Early Exit)


@dataclass
class TalamoCapture:
    """Capturas del TalamoInicial."""

    llaves_nonzero_pct: float = 0.0     # % de activaciones LLAVES no-cero
    attn_acts_mean: float = 0.0         # media de attn_proj output
    zona_acts_top5: List[Tuple[int, float]] = field(default_factory=list)  # top-5 zonas activas
    terr_distribution: List[float] = field(default_factory=list)  # [4] distribución territorial
    llaves_attn_agreement: float = 0.0  # coseno entre LLAVES y attn_proj


@dataclass
class ProbeCapture:
    """Captura completa de un forward pass."""

    talamo: Optional[TalamoCapture] = None
    niveles: List[NivelCapture] = field(default_factory=list)
    # Global
    input_len: int = 0
    total_stream_divergence: float = 0.0  # cuánto divergen los 4 streams al final


class GhidraProbe:
    """
    Instrumentación read-only del forward pass de PamparV3.

    Registra forward hooks que capturan activaciones sin gradient.
    Se activa/desactiva sin modificar el modelo.
    """

    def __init__(self, model: PamparV3):
        self._model = model
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._capture = ProbeCapture()
        self._raw: Dict[str, torch.Tensor] = {}
        self._attach()

    def _attach(self) -> None:
        """Registra forward hooks en todos los componentes."""
        # Hook en TalamoInicial
        self._hooks.append(
            self._model.talamo.register_forward_hook(self._hook_talamo)
        )

        # Hooks en cada NivelProfundo
        for i, nivel in enumerate(self._model.niveles):
            self._hooks.append(
                nivel.register_forward_hook(self._make_hook_nivel(i))
            )
            # Hook en la atención de cada nivel (captura x_attn)
            self._hooks.append(
                nivel.attn.register_forward_hook(self._make_hook_attn(i))
            )
            # Hooks en cada StreamFFN dentro del nivel
            for t, ffn in enumerate(nivel.ffns):
                self._hooks.append(
                    ffn.register_forward_hook(self._make_hook_ffn(i, t))
                )
            # Hook en LateralGate
            self._hooks.append(
                nivel.lateral.register_forward_hook(self._make_hook_lateral(i))
            )

    def detach(self) -> None:
        """Elimina todos los hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset(self) -> None:
        """Limpia capturas para un nuevo forward pass."""
        self._capture = ProbeCapture()
        self._raw.clear()

    # ── Hooks ─────────────────────────────────────────────────────────────

    def _hook_talamo(
        self,
        module: torch.nn.Module,
        input_args: Tuple,
        output: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Captura la salida del TalamoInicial."""
        terr_acts, zona_acts = output
        with torch.no_grad():
            tc = TalamoCapture()

            # Distribución territorial promedio
            terr_mean = terr_acts.mean(dim=(0, 1))  # [4]
            tc.terr_distribution = terr_mean.tolist()

            # Zona acts: top-5 zonas más activas
            zona_mean = zona_acts.mean(dim=(0, 1))  # [52]
            top_vals, top_idx = zona_mean.topk(5)
            tc.zona_acts_top5 = [
                (idx.item(), val.item()) for idx, val in zip(top_idx, top_vals)
            ]

            # LLAVES vs attn_proj agreement
            if hasattr(module, "llaves") and hasattr(module, "attn_proj"):
                x_emb = input_args[0]  # [B, L, D]
                token_ids = input_args[1]  # [B, L]
                llaves_raw = module.llaves(token_ids)  # [B, L, 52]
                attn_raw = torch.sigmoid(module.attn_proj(x_emb))  # [B, L, 52]

                # % de LLAVES no-cero
                tc.llaves_nonzero_pct = (
                    (llaves_raw > 0.01).float().mean().item() * 100
                )
                tc.attn_acts_mean = attn_raw.mean().item()

                # Coseno entre los dos promedios
                ll_flat = llaves_raw.mean(dim=(0, 1))
                at_flat = attn_raw.mean(dim=(0, 1))
                cos = torch.nn.functional.cosine_similarity(
                    ll_flat.unsqueeze(0), at_flat.unsqueeze(0)
                )
                tc.llaves_attn_agreement = cos.item()

            self._capture.talamo = tc
            self._raw["terr_acts_n0"] = terr_acts.detach().cpu()
            self._raw["zona_acts_n0"] = zona_acts.detach().cpu()

    def _make_hook_nivel(self, nivel_idx: int):
        """Factory de hook para NivelProfundo."""
        def hook(module, input_args, output):
            streams_out, terr_acts_out, conf = output
            streams_in = input_args[0]  # List[Tensor]

            with torch.no_grad():
                nc = NivelCapture(nivel_idx=nivel_idx)

                # Normas de streams entrada/salida
                nc.streams_in_norms = [
                    s.norm(dim=-1).mean().item() for s in streams_in
                ]
                nc.streams_out_norms = [
                    s.norm(dim=-1).mean().item() for s in streams_out
                ]

                # Delta = cuánto cambió cada stream
                nc.ffn_delta_norms = [
                    (streams_out[t] - streams_in[t]).norm(dim=-1).mean().item()
                    for t in range(len(streams_in))
                ]

                # Routing en este nivel
                terr_mean = terr_acts_out.mean(dim=(0, 1))
                nc.terr_acts_mean = terr_mean.tolist()
                nc.terr_acts_max_stream = terr_mean.argmax().item()

                # Exit confidence
                n_streams = len(streams_out)
                x_out = sum(
                    streams_out[t] * terr_acts_out[:, :, t:t + 1]
                    for t in range(n_streams)
                )
                per_token = torch.sigmoid(
                    module.exit_head(x_out)
                ).squeeze(-1)
                nc.exit_conf_mean = per_token.mean().item()
                nc.exit_conf_min = per_token.min().item()
                k = max(1, int(per_token.numel() * 0.10))
                nc.exit_conf_p10 = (
                    per_token.reshape(-1).topk(k, largest=False).values.mean().item()
                )

                # Lateral scales
                nc.lateral_scales = module.lateral.scale.detach().tolist()

                self._capture.niveles.append(nc)

                # Guardar raw para análisis externo
                self._raw[f"streams_out_n{nivel_idx}"] = [
                    s.detach().cpu() for s in streams_out
                ]
                self._raw[f"terr_acts_n{nivel_idx}"] = terr_acts_out.detach().cpu()

        return hook

    def _make_hook_ffn(self, nivel_idx: int, stream_idx: int):
        """Factory de hook para StreamFFN."""
        def hook(module, input_args, output):
            with torch.no_grad():
                key = f"ffn_n{nivel_idx}_s{stream_idx}"
                self._raw[key] = output.detach().cpu()
        return hook

    def _make_hook_attn(self, nivel_idx: int):
        """Factory de hook para BloqueAttn — captura x_attn en cada nivel."""
        def hook(module, input_args, output):
            with torch.no_grad():
                self._raw[f"x_attn_n{nivel_idx}"] = output.detach().cpu()
        return hook

    def _make_hook_lateral(self, nivel_idx: int):
        """Factory de hook para LateralGate."""
        def hook(module, input_args, output):
            streams_in = input_args[0]
            streams_out = output
            with torch.no_grad():
                deltas = [
                    (streams_out[t] - streams_in[t]).norm(dim=-1).mean().item()
                    for t in range(len(streams_in))
                ]
                # Actualizar el NivelCapture correspondiente
                for nc in self._capture.niveles:
                    if nc.nivel_idx == nivel_idx:
                        nc.lateral_delta_norms = deltas
                        break
        return hook

    # ── Análisis ──────────────────────────────────────────────────────────

    def report(self) -> ProbeCapture:
        """Devuelve la captura completa del último forward pass."""
        # Calcular divergencia total de streams al final
        if self._capture.niveles:
            last = self._capture.niveles[-1]
            norms = last.streams_out_norms
            if norms:
                mean_norm = sum(norms) / len(norms)
                variance = sum((n - mean_norm) ** 2 for n in norms) / len(norms)
                self._capture.total_stream_divergence = variance ** 0.5
        return self._capture

    def get_raw(self, key: str) -> Optional[torch.Tensor]:
        """Acceso a tensores raw capturados."""
        return self._raw.get(key)

    def get_all_raw_keys(self) -> List[str]:
        """Lista las claves de tensores raw disponibles."""
        return list(self._raw.keys())

    def print_summary(self, tokens: Optional[List[str]] = None) -> None:
        """Imprime resumen legible del forward pass instrumentado."""
        cap = self.report()
        print("\n" + "=" * 70)
        print("  GHIDRA PROBE — Forward Pass Analysis")
        print("=" * 70)

        # Tálamo
        if cap.talamo:
            t = cap.talamo
            print("\n── TalamoInicial ──")
            print(f"  LLAVES non-zero:    {t.llaves_nonzero_pct:.1f}%")
            print(f"  attn_proj mean:     {t.attn_acts_mean:.4f}")
            print(f"  LLAVES↔attn cosine: {t.llaves_attn_agreement:.4f}")
            print(f"  Territory dist:     ", end="")
            for i, val in enumerate(t.terr_distribution):
                print(f"{STREAM_NAMES[i]}={val:.3f}  ", end="")
            print()
            print(f"  Top-5 zonas:        ", end="")
            for idx, val in t.zona_acts_top5:
                print(f"B{idx + 1:02d}={val:.3f}  ", end="")
            print()

        # Niveles
        print("\n── Niveles de Profundidad ──")
        header = (
            f"{'Nivel':>5} │ "
            + " ".join(f"{n:>7}" for n in STREAM_NAMES)
            + " │ "
            + " ".join(f"Δ{n[:2]:>4}" for n in STREAM_NAMES)
            + " │ "
            + " ".join(f"L{n[0]:>4}" for n in STREAM_NAMES)
            + " │ Exit p10  │ Dom"
        )
        print(header)
        print("─" * len(header))

        for nc in cap.niveles:
            norms_str = " ".join(f"{v:7.1f}" for v in nc.streams_out_norms)
            delta_str = " ".join(f"{v:6.2f}" for v in nc.ffn_delta_norms)
            lat_str = " ".join(
                f"{v:6.3f}" for v in nc.lateral_delta_norms
            ) if nc.lateral_delta_norms else "  —    " * 4
            dom = STREAM_NAMES[nc.terr_acts_max_stream]
            print(
                f"  N{nc.nivel_idx:>2} │ {norms_str} │ {delta_str} │ "
                f"{lat_str} │ {nc.exit_conf_p10:8.4f}  │ {dom}"
            )

        # Lateral scales
        if cap.niveles and cap.niveles[0].lateral_scales:
            print("\n── Lateral Gate Scales (learnable) ──")
            for nc in cap.niveles:
                scales_str = " ".join(f"{v:.4f}" for v in nc.lateral_scales)
                print(f"  N{nc.nivel_idx:>2}: {scales_str}")

        # Divergencia final
        print(f"\n── Stream divergence (final): {cap.total_stream_divergence:.4f}")
        print("=" * 70)

    def token_heatmap(
        self,
        token_idx: int,
        nivel_idx: int = -1,
    ) -> Dict[str, float]:
        """
        Mapa de calor para un token específico en un nivel.

        Args:
            token_idx: posición del token en la secuencia
            nivel_idx: nivel a inspeccionar (-1 = último)

        Returns:
            Dict con activaciones territoriales y confianza del token
        """
        if nivel_idx == -1:
            nivel_idx = len(self._capture.niveles) - 1

        terr_key = f"terr_acts_n{nivel_idx}"
        terr = self._raw.get(terr_key)
        if terr is None:
            return {}

        result: Dict[str, float] = {}
        if token_idx < terr.shape[1]:
            acts = terr[0, token_idx]  # [4]
            for i, name in enumerate(STREAM_NAMES):
                result[f"terr_{name}"] = acts[i].item()

        streams_key = f"streams_out_n{nivel_idx}"
        streams = self._raw.get(streams_key)
        if streams and token_idx < streams[0].shape[1]:
            for i, name in enumerate(STREAM_NAMES):
                result[f"norm_{name}"] = (
                    streams[i][0, token_idx].norm().item()
                )

        return result

    def routing_trajectory(
        self,
        token_idx: int,
    ) -> List[List[float]]:
        """
        Trayectoria de routing de un token a través de todos los niveles.

        Returns:
            Lista de [4] activaciones territoriales por nivel (incluyendo N0)
        """
        trajectory: List[List[float]] = []

        # N0 (TalamoInicial)
        terr_n0 = self._raw.get("terr_acts_n0")
        if terr_n0 is not None and token_idx < terr_n0.shape[1]:
            trajectory.append(terr_n0[0, token_idx].tolist())

        # N1-N4
        for i in range(len(self._capture.niveles)):
            terr = self._raw.get(f"terr_acts_n{i}")
            if terr is not None and token_idx < terr.shape[1]:
                trajectory.append(terr[0, token_idx].tolist())

        return trajectory
