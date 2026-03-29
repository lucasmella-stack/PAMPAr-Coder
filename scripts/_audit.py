"""Audit script: verify all LLAVES corrections are consistent."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pampar.coder.v3.llaves import clasificar_token
from pampar.coder.v3.zonas import (
    ZONA_TERRITORIO,
    ZONAS,
    ZONAS_POR_TERRITORIO,
    Territorio,
    Zona,
)

# 1. Overrides activos
print("=== OVERRIDES ACTIVOS ===")
for z in Zona:
    natural = 0 if z.value <= 15 else 1 if z.value <= 30 else 2 if z.value <= 42 else 3
    actual = ZONA_TERRITORIO[z].value
    if natural != actual:
        print(
            f"  {z.name}: natural={Territorio(natural).name} -> override={Territorio(actual).name}"
        )

# 2. B03 y B52
print(f"\nB03_KW_IMPORT: {ZONAS[Zona.B03_KW_IMPORT]}")
print(f"B52_PATTERN_CALL: {ZONAS[Zona.B52_PATTERN_CALL]}")
print(f"B11_DELIM_PAREN: {ZONAS[Zona.B11_DELIM_PAREN]}")
print(f"B07_KW_EXCEPT: {ZONAS[Zona.B07_KW_EXCEPT]}")
print(f"B09_KW_MOD: {ZONAS[Zona.B09_KW_MOD]}")

# 3. Clasificación de tokens clave
print("\n=== CLASIFICACION TOKENS CLAVE ===")
tokens = [
    "from",
    "import",
    "try",
    "except",
    "finally",
    "raise",
    "staticmethod",
    "classmethod",
    "property",
    "(",
    ")",
    "('",
    '("',
    "')",
    '")',
    "=",
    "+=",
    "with",
    "as",
    ".",
    "or",
    "!=",
]
for t in tokens:
    zona, conf = clasificar_token(t)
    terr = ZONA_TERRITORIO[zona]
    print(f"  {t:20s} -> zona={zona.name:20s} terr={terr.name:12s} conf={conf:.0%}")

# 4. Con prefijo SentencePiece
print("\n=== CON PREFIJO SENTENCEPIECE ===")
sp_tokens = [
    "\u2581from",
    "\u2581import",
    "\u2581try",
    "\u2581except",
    "\u2581=",
    "\u2581with",
    "\u2581staticmethod",
    "('",
    "')",
    "\u2581or",
    "\u2581!=",
]
for t in sp_tokens:
    zona, conf = clasificar_token(t)
    terr = ZONA_TERRITORIO[zona]
    print(f"  {t:20s} -> zona={zona.name:20s} terr={terr.name:12s} conf={conf:.0%}")

# 5. ZONAS_POR_TERRITORIO
print("\n=== ZONAS POR TERRITORIO ===")
for t in Territorio:
    zonas = ZONAS_POR_TERRITORIO[t]
    print(f"  {t.name:12s}: {len(zonas)} zonas")
    overridden = [
        z
        for z in zonas
        if (0 if z.value <= 15 else 1 if z.value <= 30 else 2 if z.value <= 42 else 3)
        != t.value
    ]
    if overridden:
        print(f"    overrides: {[z.name for z in overridden]}")

# 6. Duplicates check: no token appears in two zonas
print("\n=== DUPLICATES CHECK ===")
seen: dict[str, list[str]] = {}
for zona, patrones in ZONAS.items():
    for tok in patrones:
        if tok in seen:
            seen[tok].append(zona.name)
        else:
            seen[tok] = [zona.name]
dupes = {k: v for k, v in seen.items() if len(v) > 1}
if dupes:
    print(f"  DUPLICADOS ENCONTRADOS: {dupes}")
else:
    print("  Sin duplicados entre zonas")

# 7. agregar_zonas_a_territorios uses hardcoded ranges — check alignment
print("\n=== HARDCODED RANGES VS OVERRIDES ===")
print("  agregar_zonas_a_territorios usa rangos fijos:")
print("    SINT: B01-B15 (idx 0-14)")
print("    SEMA: B16-B30 (idx 15-29)")
print("    LOGI: B31-B42 (idx 30-41)")
print("    ESTR: B43-B52 (idx 42-51)")
print()
print("  NOTA: Esta funcion agrega zona_acts -> terr_acts dentro del MODELO.")
print("  Los overrides de ZONA_TERRITORIO solo afectan al territory_table")
print("  usado como GROUND TRUTH en brain_scanner y ghidra_trainer loss.")
print("  El modelo sigue mapeando zonas a territorios por RANGO FIJO.")
print("  Esto es CORRECTO: el override cambia QUE se considera correcto,")
print("  no como el modelo internamente rutea.")
