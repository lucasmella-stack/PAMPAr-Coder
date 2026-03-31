"""Análisis rápido de sesiones del Classroom."""

import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "sessions/classroom_20260330_223816.jsonl"

with open(path, "r", encoding="utf-8") as f:
    lessons = [json.loads(line) for line in f]

total = len(lessons)
correct_total = sum(1 for l in lessons if l["correct"])
print(f"Total lecciones: {total}")
print(f"Correctas: {correct_total} / {total} ({100 * correct_total / total:.1f}%)")
print()

# Loss por bloques de 20
print("=== Progresión por bloques de 20 ===")
for i in range(0, total, 20):
    chunk = lessons[i : i + 20]
    avg_loss = sum(l["loss"] for l in chunk) / len(chunk)
    avg_brain = sum(l["brain_score"] for l in chunk) / len(chunk)
    correct = sum(1 for l in chunk if l["correct"])
    concepts = set(l.get("problem", "")[:30] for l in chunk)
    print(
        f"  L{i + 1:3d}-{i + len(chunk):3d}: "
        f"loss={avg_loss:.3f}  brain={avg_brain:.2%}  correct={correct}/{len(chunk)}"
    )

print()
print("=== Primeras 5 respuestas ===")
for l in lessons[:5]:
    ans = (l["student_answer"] or "(vacío)")[:100]
    print(f"  L{l['lesson_id']:3d}: {ans}")

print()
print("=== Últimas 5 respuestas ===")
for l in lessons[-5:]:
    ans = (l["student_answer"] or "(vacío)")[:100]
    print(f"  L{l['lesson_id']:3d}: {ans}")

# Conceptos vistos
concepts_seen = {}
for l in lessons:
    # Extraer concepto del feedback o problema
    fb = l.get("feedback", "")
    if l["correct"]:
        concepts_seen[l["lesson_id"]] = "correct"

print()
print(f"=== Loss primera vs última lección ===")
print(f"  Primera: loss={lessons[0]['loss']:.4f}")
print(f"  Última:  loss={lessons[-1]['loss']:.4f}")
print(
    f"  Min:     loss={min(l['loss'] for l in lessons):.4f} (lección {min(lessons, key=lambda x: (
            x['loss']
        ))['lesson_id']})"
)
print(
    f"  Max:     loss={max(l['loss'] for l in lessons):.4f} (lección {max(lessons, key=lambda x: (
            x['loss']
        ))['lesson_id']})"
)
