"""classroom_persistence.py — Guardado de checkpoints, sesiones y grabaciones."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from classroom_curriculum import ClassroomConfig
    from classroom_memory import LessonResult


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: "ClassroomConfig",
    lesson_count: int,
    current_level: int,
    total_correct: int,
) -> str:
    """Guarda checkpoint del modelo.

    Returns:
        Ruta del checkpoint guardado.
    """
    project_root = Path(__file__).parent.parent
    ckpt_path = project_root / config.checkpoint_out
    torch.save(
        {
            "modelo": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "paso_global": lesson_count,
            "config": asdict(config),
            "curriculum_level": current_level,
            "accuracy": total_correct / max(1, lesson_count),
        },
        str(ckpt_path),
    )
    return str(ckpt_path)


def save_session(session_log: list["LessonResult"]) -> str:
    """Guarda la sesión completa como JSONL.

    Returns:
        Ruta del archivo guardado.
    """
    project_root = Path(__file__).parent.parent
    ts = time.strftime("%Y%m%d_%H%M%S")
    session_path = project_root / f"sessions/classroom_{ts}.jsonl"
    session_path.parent.mkdir(parents=True, exist_ok=True)

    with open(session_path, "w", encoding="utf-8") as f:
        for r in session_log:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    return str(session_path)


def save_recording(
    events: list[dict],
    recording_start: float,
    config: "ClassroomConfig",
    lesson_count: int,
    total_correct: int,
    current_level: int,
) -> str:
    """Guarda la grabación completa de eventos como HTML reproducible.

    Returns:
        Ruta del archivo HTML o cadena vacía si no hay eventos.
    """
    if not events:
        return ""

    project_root = Path(__file__).parent.parent
    ts = time.strftime("%Y%m%d_%H%M%S")
    recording_dir = project_root / "sessions"
    recording_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "model": "PamparV3 (108M)",
        "teacher_backend": config.teacher_backend,
        "teacher_model": config.teacher_model,
        "start_time": time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(recording_start),
        ),
        "duration_s": round(time.time() - recording_start, 1)
        if recording_start
        else 0,
        "total_lessons": lesson_count,
        "accuracy": round(total_correct / max(1, lesson_count), 4),
        "final_level": current_level,
        "ewc_lambda": config.ewc_lambda,
        "lr_base": config.lr_base,
    }

    replay_template = Path(__file__).parent / "classroom_replay.html"
    if replay_template.exists():
        template = replay_template.read_text(encoding="utf-8")
    else:
        template = "<html><body><pre>No replay template found</pre></body></html>"

    recording_data = json.dumps(
        {"meta": meta, "events": events},
        ensure_ascii=False,
    )

    html = template.replace(
        "/*__RECORDING_DATA__*/",
        f"window.__RECORDING__ = {recording_data};",
    )

    out_path = recording_dir / f"classroom_{ts}.html"
    out_path.write_text(html, encoding="utf-8")

    return str(out_path)
