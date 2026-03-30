"""classroom_events.py — Formateo e impresión de eventos del Classroom."""

from __future__ import annotations


def format_event_to_console(event_type: str, data: str | dict) -> None:
    """Imprime un evento del classroom en consola con formato legible."""
    d = data if isinstance(data, dict) else {}

    _FORMATTERS.get(event_type, _noop)(event_type, data, d)


def _noop(_event_type: str, _data: str | dict, _d: dict) -> None:
    pass


def _fmt_system(_et: str, data: str | dict, _d: dict) -> None:
    print(f"  🏫 {data}")


def _fmt_lesson_start(_et: str, _data: str | dict, d: dict) -> None:
    concept = d.get("concept", "")
    print(
        f"\n  ═══ Lección {d.get('lesson_id', '?')} — "
        f"{d.get('level_name', '')} [{concept}] "
        f"(Nivel {d.get('level', '?')}) ═══"
    )
    print(f"  📝 {d.get('problem', '')[:100]}")


def _fmt_mentor_explain(_et: str, _data: str | dict, d: dict) -> None:
    print(f"  📖 Mentor explica: {d.get('explain', '')[:120]}")


def _fmt_mentor_example(_et: str, _data: str | dict, d: dict) -> None:
    example = d.get("example", "")
    lines = example.split("\n")
    preview = lines[0][:80] if lines else ""
    print(f"  💻 Mentor ejemplo: {preview}{'...' if len(lines) > 1 else ''}")


def _fmt_student_answer(_et: str, _data: str | dict, d: dict) -> None:
    print(f"  🧑‍🎓 Alumno: {d.get('answer', '')[:100]}")


def _fmt_teacher_feedback(_et: str, _data: str | dict, d: dict) -> None:
    icon = "✅" if d.get("correct") else "❌"
    print(f"  👨‍🏫 Profesor: {icon} {d.get('feedback', '')[:100]}")


def _fmt_lesson_complete(_et: str, _data: str | dict, d: dict) -> None:
    concept = d.get("concept", "")
    print(
        f"  📊 Loss: {d.get('loss', 0):.4f} | "
        f"EWC: {d.get('ewc_penalty', 0):.6f} | "
        f"Brain: {d.get('brain_score', 0):.2%} | "
        f"Acc: {d.get('accuracy', 0):.1%} | "
        f"Replay: {d.get('replay_size', 0)} | "
        f"Concepto: {concept}"
    )


def _fmt_level_up(_et: str, _data: str | dict, d: dict) -> None:
    print(f"\n  🎉 ¡NIVEL UP! → Nivel {d.get('new_level', '?')}: {d.get('nombre', '')}")


def _fmt_checkpoint(_et: str, _data: str | dict, d: dict) -> None:
    print(f"  💾 Checkpoint guardado: lección {d.get('lesson', '?')}")


def _fmt_bio_update(_et: str, _data: str | dict, d: dict) -> None:
    parts = [
        f"DA={d.get('dopamine', 0):.2f}",
        f"NE={d.get('norepinephrine', 0):.2f}",
        f"LR×{d.get('lr_factor', 1):.2f}",
    ]
    if d.get("ltp_applied"):
        parts.append("LTP!")
    if d.get("sleep_triggered"):
        parts.append(f"SLEEP(loss={d.get('sleep_loss', 0):.3f})")
    if d.get("adapters_total", 0) > 0:
        parts.append(f"LoRA={d.get('adapters_total', 0)}")
    if d.get("pruned"):
        parts.append("PRUNED")
    print(f"  🧠 Bio: {' | '.join(parts)}")


def _fmt_error(_et: str, data: str | dict, _d: dict) -> None:
    print(f"  ❗ {data}")


_FORMATTERS: dict[str, object] = {
    "system": _fmt_system,
    "lesson_start": _fmt_lesson_start,
    "mentor_explain": _fmt_mentor_explain,
    "mentor_example": _fmt_mentor_example,
    "student_answer": _fmt_student_answer,
    "teacher_feedback": _fmt_teacher_feedback,
    "lesson_complete": _fmt_lesson_complete,
    "level_up": _fmt_level_up,
    "checkpoint": _fmt_checkpoint,
    "bio_update": _fmt_bio_update,
    "error": _fmt_error,
}
