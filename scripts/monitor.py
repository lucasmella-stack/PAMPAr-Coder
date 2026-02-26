#!/usr/bin/env python3
"""Monitor en vivo del entrenamiento PAMPAr.

Uso:
  python scripts/monitor.py
  python scripts/monitor.py --cada 10   # actualizar cada 10s
"""
import argparse
import json
import os
import time
from pathlib import Path

# Colores ANSI
B  = "\033[1m"
AZ = "\033[94m"
VE = "\033[92m"
AM = "\033[93m"
RO = "\033[91m"
GR = "\033[90m"
RE = "\033[0m"


def limpiar():
    os.system("cls" if os.name == "nt" else "clear")


def leer_estado(ruta: Path) -> dict:
    try:
        return json.loads(ruta.read_text(encoding="utf-8"))
    except Exception:
        return {}


def checkpoint_info(ruta: Path) -> tuple[str, str]:
    if not ruta.exists():
        return "no encontrado", "?"
    stat = ruta.stat()
    ts = time.strftime("%H:%M:%S", time.localtime(stat.st_mtime))
    mb = stat.st_size / 1e6
    return f"guardado a las {ts}  ({mb:.1f} MB)", ts


def render(estado_path: Path, ckpt_path: Path) -> None:
    d = leer_estado(estado_path)
    temas = d.get("temas", {})
    nivel = d.get("nivel_actual", "?")
    sesiones_total = d.get("sesiones_totales", 0)
    n_dominados = d.get("temas_dominados", 0)  # es un int
    paso = d.get("paso_global", "?")

    ck_str, _ = checkpoint_info(ckpt_path)

    ahora = time.strftime("%H:%M:%S")
    print(f"{B}{AZ}═══ PAMPAr Monitor — {ahora} ══════════════════════════════{RE}")
    paso_fmt = f"{paso:,}" if isinstance(paso, int) else str(paso)
    print(f"  Paso global : {B}{paso_fmt}{RE}     Nivel: {nivel}/6     Sesiones: {sesiones_total}")
    print(f"  Checkpoint  : {GR}{ck_str}{RE}")
    print()

    if not temas:
        print(f"  {AM}Sin temas cargados aún{RE}")
        return

    # Temas con mayor loss (más difíciles = más trabajo pendiente)
    dominados = [(k, v) for k, v in temas.items() if v.get("dominado")]
    activos   = [(k, v) for k, v in temas.items() if not v.get("dominado")]
    activos.sort(key=lambda x: (x[1].get("historial_loss") or [0])[-1], reverse=True)

    print(f"  {B}Temas con mayor loss (más a trabajar):{RE}")
    for tema, v in activos[:12]:
        hist = v.get("historial_loss") or [0]
        loss = hist[-1]
        n    = v.get("nivel_dificultad", 0)
        ses  = v.get("n_sesiones", 0)
        cur  = v.get("curiosidad", 0)
        bar  = "█" * min(int(loss), 8) + "░" * max(0, 8 - int(loss))
        color = RO if loss > 3 else AM if loss > 1 else VE
        print(f"    {color}{bar}{RE}  {tema:<30}  loss={loss:.3f}  cur={cur:.3f}  sesiones={ses}")

    print()
    n_dom = n_dominados if n_dominados else len(dominados)
    n_tot = len(temas)
    pct = n_dom / n_tot * 100 if n_tot else 0
    bloques = 30
    llenos = int(pct / 100 * bloques)
    barra_dom = VE + "█" * llenos + RE + GR + "░" * (bloques - llenos) + RE
    print(f"  Dominados: {VE}{n_dom}/{n_tot}{RE}  [{barra_dom}]  {pct:.0f}%")

    if dominados:
        print(f"  {GR}Temas dominados: {', '.join(k for k, _ in dominados[:8])}{RE}")

    print(f"\n{GR}  (Ctrl+C para salir){RE}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cada", type=int, default=5, help="Segundos entre actualizaciones")
    parser.add_argument(
        "--estado", type=Path,
        default=Path("checkpoints/curiosidad_estado.json"),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/pampar_v2_best.pt"),
    )
    args = parser.parse_args()

    try:
        first = True
        while True:
            if not first:
                limpiar()
            first = False
            render(args.estado, args.checkpoint)
            if args.cada <= 0:
                break
            time.sleep(args.cada)
    except KeyboardInterrupt:
        print("\nMonitor detenido.")


if __name__ == "__main__":
    main()
