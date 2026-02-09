#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Descargar texto español masivo y re-entrenar tokenizer 48K.

Usa Wikipedia español de HuggingFace (gratuito, sin auth).
Luego re-entrena SentencePiece con corpus código + español.
"""
import os
import sys
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
TOKENIZER_DIR = DATA_DIR / "tokenizer"
SPANISH_FILE = TOKENIZER_DIR / "spanish_wiki.txt"
CORPUS_FILE = TOKENIZER_DIR / "corpus_48k.txt"
OUTPUT_PREFIX = str(TOKENIZER_DIR / "pampar_48k")

TARGET_ES_MB = 500  # Queremos ~500MB de español


def download_spanish_wiki():
    """Descargar Wikipedia español desde HuggingFace datasets."""
    from datasets import load_dataset
    
    print("📥 Descargando Wikipedia ES desde HuggingFace...")
    print("   Dataset: wikimedia/wikipedia, 20231101.es")
    print("   Esto puede tardar 3-10 minutos...\n")
    
    ds = load_dataset(
        "wikimedia/wikipedia", 
        "20231101.es",
        split="train",
        streaming=True,
    )
    
    count = 0
    with open(SPANISH_FILE, "w", encoding="utf-8") as f:
        for example in ds:
            text = example.get("text", "")
            if len(text) > 200:  # Solo artículos sustanciales
                f.write(text.strip() + "\n")
                count += 1
                
                if count % 50000 == 0:
                    size_mb = SPANISH_FILE.stat().st_size / 1024**2
                    print(f"   {count:,} artículos ({size_mb:.0f} MB)")
                    
                    if size_mb >= TARGET_ES_MB:
                        print(f"   Alcanzado objetivo de {TARGET_ES_MB} MB")
                        break
    
    size_mb = SPANISH_FILE.stat().st_size / 1024**2
    print(f"\n✅ Wikipedia ES: {count:,} artículos, {size_mb:.0f} MB")
    return count


def rebuild_corpus():
    """Reconstruir corpus combinando código + español + código ES."""
    print("\n📦 Reconstruyendo corpus bilingüe...")
    
    # 1. Código existente (nuestros JSONL)
    code_files = [
        DATA_DIR / "code" / "github_code.jsonl",
        DATA_DIR / "code" / "train_massive.jsonl",
        DATA_DIR / "code" / "train.jsonl",
        DATA_DIR / "distillation" / "codealpaca_20k.jsonl",
        DATA_DIR / "distillation" / "evol_instruct_code_80k.jsonl",
        DATA_DIR / "distillation" / "codeexercises_python.jsonl",
        DATA_DIR / "distillation" / "distillation_data.jsonl",
    ]
    
    total = 0
    with open(CORPUS_FILE, "w", encoding="utf-8") as out:
        # Código
        print("   Código:")
        for jsonl_path in code_files:
            if not jsonl_path.exists():
                continue
            size_mb = jsonl_path.stat().st_size / (1024 * 1024)
            file_count = 0
            with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get("text", "")
                        if len(text) > 50:
                            out.write(text.strip() + "\n")
                            total += 1
                            file_count += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
            print(f"     {jsonl_path.name}: {file_count:,} docs ({size_mb:.1f} MB)")
        
        # Español de Wikipedia
        if SPANISH_FILE.exists():
            print(f"   Wikipedia ES:")
            es_count = 0
            with open(SPANISH_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if len(line.strip()) > 50:
                        out.write(line)
                        total += 1
                        es_count += 1
            print(f"     {es_count:,} artículos")
        
        # Español sintético (términos de programación)
        print("   Código español sintético:")
        es_code = generate_spanish_code()
        for line in es_code:
            out.write(line + "\n")
            total += 1
        print(f"     {len(es_code):,} líneas")
        
        # Corpus existente anterior
        old_corpus = TOKENIZER_DIR / "corpus.txt"
        if old_corpus.exists():
            print(f"   Corpus antiguo:")
            old_count = 0
            with open(old_corpus, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if len(line.strip()) > 20:
                        out.write(line)
                        total += 1
                        old_count += 1
            print(f"     {old_count:,} docs")
    
    corpus_mb = CORPUS_FILE.stat().st_size / 1024**2
    print(f"\n📊 Corpus final: {total:,} docs, {corpus_mb:.0f} MB")
    
    # Verify Spanish content
    es_lines = 0
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if any(c in line for c in "áéíóúñü"):
                es_lines += 1
    es_pct = es_lines * 100 / total if total else 0
    print(f"   Líneas con español: {es_lines:,} ({es_pct:.1f}%)")
    
    return total


def generate_spanish_code():
    """Genera corpus sustancial de código con español."""
    lines = []
    
    # Términos que DEBEN ser tokens propios
    terms = [
        "función", "parámetro", "argumento", "retorno", "resultado",
        "variable", "constante", "método", "clase", "objeto",
        "archivo", "directorio", "configuración", "conexión", "autenticación",
        "usuario", "contraseña", "validación", "verificación", "búsqueda",
        "algoritmo", "estructura", "índice", "cálculo", "número",
        "cadena", "entero", "flotante", "booleano", "tamaño",
        "longitud", "cantidad", "máximo", "mínimo", "promedio",
        "iteración", "condición", "comparación", "operación", "excepción",
        "implementación", "definición", "inicialización", "herencia", "abstracción",
        "compilación", "ejecución", "depuración", "prueba", "documentación",
        "capa", "neurona", "peso", "activación", "gradiente",
        "optimizador", "pérdida", "precisión", "época", "entrenamiento",
        "inferencia", "predicción", "modelo", "vocabulario", "tokenización",
        "normalización", "regularización", "aprendizaje", "atención", "transformador",
    ]
    
    templates = [
        'def {0}(self, {1}):\n    """Calcula el {2} del {1} proporcionado."""\n    return self._{1}',
        '# {0}: se encarga de procesar el {1} y calcular el {2}',
        'raise ValueError("El {0} no puede estar vacío")',
        'logger.info(f"Procesando {0} con {1}: {{valor}}")',
        '"""\n{0}.\n\nParámetros:\n    {1}: El valor de {2} a procesar.\n\nRetorna:\n    El {2} calculado.\n"""',
        'self.{0} = {1}  # {2} del componente',
        '# TODO: implementar {0} para mejorar el {1}',
        'if not self.{0}:\n    raise RuntimeError("Falta el {1} para la {2}")',
        'print(f"Error en {0}: el {1} no es válido para {2}")',
        'assert isinstance({0}, {1}), f"Se esperaba {1}, se recibió {{type({0})}}"',
    ]
    
    import random
    random.seed(42)
    
    for _ in range(2000):
        random.shuffle(terms)
        for i in range(0, len(terms) - 2, 3):
            for t in templates:
                try:
                    lines.append(t.format(terms[i], terms[i+1], terms[i+2]))
                except (IndexError, KeyError):
                    pass
    
    # Terms repeated standalone
    for _ in range(2000):
        for term in terms:
            lines.append(term)
    
    return lines


def retrain_tokenizer():
    """Re-entrenar SentencePiece BPE con corpus bilingüe."""
    import sentencepiece as spm
    
    corpus_mb = CORPUS_FILE.stat().st_size / 1024**2
    print(f"\n🔧 Re-entrenando tokenizer BPE...")
    print(f"   Corpus: {corpus_mb:.0f} MB")
    print(f"   Vocab:  48,000")
    print(f"   Esto puede tardar 10-30 minutos...\n")
    
    spm.SentencePieceTrainer.train(
        input=str(CORPUS_FILE),
        model_prefix=OUTPUT_PREFIX,
        vocab_size=48000,
        model_type="bpe",
        pad_id=0, eos_id=1, bos_id=2, unk_id=3,
        character_coverage=0.9999,
        num_threads=os.cpu_count() or 4,
        train_extremely_large_corpus=True,
        max_sentence_length=16384,
        byte_fallback=True,
        normalization_rule_name="identity",
        split_digits=True,
        user_defined_symbols=[
            "    ", "        ", "            ", "\t",
            "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=",
            "->", "=>", "::", "//", "**", "&&", "||",
            "...", "..=", '"""', "'''", "```",
        ],
        control_symbols=["<pad>", "<mask>"],
    )
    
    print("✅ Tokenizer re-entrenado!")


def verify():
    """Verificar tokenizer."""
    import sentencepiece as spm
    
    sp = spm.SentencePieceProcessor()
    sp.load(f"{OUTPUT_PREFIX}.model")
    
    print(f"\n🧪 Verificación del tokenizer")
    print(f"   Vocab: {sp.get_piece_size():,}")
    
    # Test español
    tests = {
        "función":    "función",
        "número":     "número",
        "tamaño":     "tamaño",
        "cálculo":    "cálculo",
        "también":    "también",
        "España":     "España",
        "programación": "programación",
    }
    
    print(f"\n   Palabras españolas:")
    all_ok = True
    for name, word in tests.items():
        enc = sp.encode(word, out_type=str)
        has_bytes = any("<0x" in t for t in enc)
        ok = "✅" if not has_bytes else "❌"
        if has_bytes:
            all_ok = False
        print(f"   {ok} {word:20s} -> {enc}")
    
    # Eficiencia
    tests2 = {
        "Python": "def calcular_promedio(numeros):\n    return sum(numeros) / len(numeros)",
        "Español": "La función calcula el promedio de una lista de números flotantes.",
        "TypeScript": "const resultado: number = await procesarDatos(entrada);",
        "SQL": "SELECT nombre, COUNT(*) AS total FROM usuarios GROUP BY nombre",
    }
    
    print(f"\n   Eficiencia (tok/char):")
    for name, text in tests2.items():
        ids = sp.encode(text)
        ratio = len(ids) / len(text)
        print(f"   {name:15s}: {ratio:.3f} ({len(ids)} tok / {len(text)} chars)")
    
    # Count accent tokens in vocab
    accent_count = sum(1 for i in range(sp.get_piece_size()) 
                      if any(c in sp.id_to_piece(i) for c in "áéíóúñüÁÉÍÓÚÑÜ"))
    print(f"\n   Tokens con acentos/ñ: {accent_count:,} de {sp.get_piece_size():,}")
    
    if all_ok:
        print("\n✅ El tokenizer maneja español correctamente!")
    else:
        print("\n⚠️  Algunas palabras todavía usan byte fallback")


if __name__ == "__main__":
    # 1. Descargar Wikipedia ES
    download_spanish_wiki()
    
    # 2. Reconstruir corpus bilingüe
    rebuild_corpus()
    
    # 3. Re-entrenar tokenizer
    retrain_tokenizer()
    
    # 4. Verificar
    verify()
