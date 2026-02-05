# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Test de LLAVES para PAMPAr-Coder.

Verifica que los tokens de código se clasifican correctamente
en los territorios apropiados.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sentencepiece as spm
from pampar.coder.llaves_codigo import LlavesCodigo, TipoTerritorioCoder


def test_con_codigo_real():
    """Prueba clasificación con código real."""
    tokenizer = spm.SentencePieceProcessor(
        model_file=str(Path(__file__).parent.parent / "data/tokenizer/code_tokenizer.model")
    )
    llaves = LlavesCodigo()
    
    code = """def calculate(x, y):
    if x > 0:
        return x + y
    else:
        return 0
"""
    
    print("Código de prueba:")
    print(code)
    print("-" * 70)
    print(f"{'ID':>5} | {'Token':15} | {'Territorio':12} | {'Peso':4} | Activaciones")
    print("-" * 70)
    
    token_ids = tokenizer.encode(code)
    
    stats = {t: 0 for t in TipoTerritorioCoder}
    
    for tid in token_ids:
        piece = tokenizer.id_to_piece(tid)
        acts = llaves.activar_territorios(piece)
        principal = max(acts, key=acts.get)
        peso = acts[principal]
        
        stats[principal] += 1
        
        # Mostrar activaciones
        activos = " | ".join(f"{t.name[:4]}:{v:.1f}" for t, v in acts.items() if v > 0)
        print(f"{tid:5} | {repr(piece):15} | {principal.name:12} | {peso:.1f}  | [{activos}]")
    
    print("-" * 70)
    print("\nResumen de distribución en este código:")
    total = sum(stats.values())
    for t, count in stats.items():
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {t.name:12} {bar:30} {pct:5.1f}% ({count}/{total})")


def test_tokens_especificos():
    """Prueba tokens específicos de código."""
    llaves = LlavesCodigo()
    
    print("\n" + "=" * 70)
    print("Test de tokens específicos de código")
    print("=" * 70)
    
    grupos = {
        "Keywords Python": ["def", "class", "return", "import", "from", "as", "with"],
        "Keywords JS": ["function", "const", "let", "var", "export", "default"],
        "Keywords Rust": ["fn", "let", "mut", "struct", "impl", "trait", "pub"],
        "Control Flow": ["if", "else", "elif", "for", "while", "try", "except", "match"],
        "Operadores": ["+", "-", "*", "/", "==", "!=", "&&", "||", "->", "=>"],
        "Brackets": ["(", ")", "[", "]", "{", "}"],
        "Delimitadores": [":", ";", ",", "@", "#"],
        "Tipos": ["int", "str", "float", "bool", "Vec", "Option", "String"],
        "Identificadores": ["myVar", "calculate_sum", "userId", "getData"],
        "Con prefijo SPM": ["▁def", "▁if", "▁class", "▁return", "▁myVar"],
    }
    
    for grupo, tokens in grupos.items():
        print(f"\n{grupo}:")
        for token in tokens:
            principal, peso = llaves.clasificar_token(token)
            print(f"  {repr(token):15} -> {principal.name:12} ({peso:.1f})")


if __name__ == "__main__":
    test_con_codigo_real()
    test_tokens_especificos()
