# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Preparar tokenizer BPE especializado para código.

Este tokenizer está optimizado para:
- Keywords de Python, JavaScript, Rust, Go
- Operadores y delimitadores como tokens únicos
- Indentación preservada
- Strings y comentarios
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter
import re

# Intentar usar sentencepiece, si no tokenizers de HuggingFace
try:
    import sentencepiece as spm
    USE_SPM = True
except ImportError:
    USE_SPM = False
    print("⚠️ sentencepiece no encontrado, usando tokenizers de HuggingFace")
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
        from tokenizers.normalizers import NFKC
    except ImportError:
        print("❌ Instala: pip install sentencepiece tokenizers")
        sys.exit(1)


# =============================================================================
# Tokens especiales para código
# =============================================================================

SPECIAL_TOKENS = [
    "<pad>",      # Padding
    "<unk>",      # Unknown
    "<bos>",      # Beginning of sequence
    "<eos>",      # End of sequence
    "<mask>",     # Masked token (para MLM)
    "<sep>",      # Separator
    # Código específico
    "<indent>",   # Indentación
    "<dedent>",   # Des-indentación
    "<newline>",  # Nueva línea
    "<comment>",  # Inicio de comentario
]

# Keywords que queremos como tokens únicos
CODE_KEYWORDS = {
    # Python
    'def', 'class', 'return', 'yield', 'async', 'await', 'import', 'from',
    'if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally',
    'with', 'as', 'lambda', 'pass', 'break', 'continue', 'raise', 'assert',
    'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is', 'global',
    # JavaScript/TypeScript
    'function', 'const', 'let', 'var', 'export', 'default', 'extends',
    'super', 'this', 'new', 'delete', 'typeof', 'instanceof', 'void',
    'true', 'false', 'null', 'undefined', 'throw', 'catch',
    # Rust
    'fn', 'let', 'mut', 'const', 'static', 'struct', 'enum', 'impl',
    'trait', 'type', 'where', 'pub', 'crate', 'mod', 'use', 'self',
    'match', 'loop', 'move', 'unsafe', 'dyn', 'ref',
    # Go
    'func', 'package', 'import', 'type', 'struct', 'interface', 'map',
    'chan', 'go', 'defer', 'select', 'case', 'range', 'fallthrough',
}

# Operadores como tokens únicos
CODE_OPERATORS = [
    # Asignación
    '=', '+=', '-=', '*=', '/=', '%=', '**=', '//=',
    '&=', '|=', '^=', '>>=', '<<=',
    # Comparación
    '==', '!=', '<', '>', '<=', '>=',
    # Aritméticos
    '+', '-', '*', '/', '%', '**', '//',
    # Lógicos
    '&&', '||', '!', 'and', 'or', 'not',
    # Bitwise
    '&', '|', '^', '~', '<<', '>>',
    # Especiales
    '->', '=>', '::', '..', '...', '?.', '??',
    # Delimitadores
    '(', ')', '[', ']', '{', '}',
    ',', ';', ':', '.', '@', '#',
    # Strings
    '"', "'", '`', '"""', "'''",
]


def create_sample_code():
    """Genera código de ejemplo para entrenar el tokenizer."""
    samples = []
    
    # Python samples
    samples.append('''
def fibonacci(n: int) -> int:
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        return x + y
    
    async def fetch_data(self, url):
        async with aiohttp.get(url) as response:
            return await response.json()

for i in range(10):
    print(f"Number: {i}")

try:
    result = dangerous_operation()
except ValueError as e:
    print(f"Error: {e}")
finally:
    cleanup()
''')
    
    # JavaScript samples
    samples.append('''
function quickSort(arr) {
    if (arr.length <= 1) return arr;
    const pivot = arr[0];
    const left = arr.slice(1).filter(x => x < pivot);
    const right = arr.slice(1).filter(x => x >= pivot);
    return [...quickSort(left), pivot, ...quickSort(right)];
}

const fetchData = async (url) => {
    try {
        const response = await fetch(url);
        return await response.json();
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
};

class Component extends React.Component {
    constructor(props) {
        super(props);
        this.state = { count: 0 };
    }
    
    render() {
        return <div>{this.state.count}</div>;
    }
}
''')
    
    # Rust samples
    samples.append('''
fn main() {
    let mut vec: Vec<i32> = Vec::new();
    
    for i in 0..10 {
        vec.push(i * 2);
    }
    
    let sum: i32 = vec.iter().sum();
    println!("Sum: {}", sum);
}

struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
    
    fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}

match result {
    Ok(value) => println!("Success: {}", value),
    Err(e) => eprintln!("Error: {}", e),
}
''')
    
    return '\n'.join(samples)


def download_code_dataset(output_dir: Path, max_size_mb: int = 100):
    """
    Descarga un dataset pequeño de código para el tokenizer.
    Usa codeparrot/github-code-clean (subset pequeño).
    """
    print("📥 Descargando dataset de código...")
    
    try:
        from datasets import load_dataset
        
        # Cargar subset pequeño
        ds = load_dataset(
            "codeparrot/github-code-clean",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        
        # Filtrar por lenguajes que nos interesan
        languages = {'Python', 'JavaScript', 'TypeScript', 'Rust', 'Go'}
        
        output_file = output_dir / "code_corpus.txt"
        total_bytes = 0
        max_bytes = max_size_mb * 1024 * 1024
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in ds:
                if sample.get('language') in languages:
                    code = sample.get('code', '')
                    if len(code) > 50:  # Filtrar snippets muy cortos
                        f.write(code)
                        f.write('\n\n')
                        total_bytes += len(code.encode('utf-8'))
                        
                        if total_bytes >= max_bytes:
                            break
                        
                        if total_bytes % (10 * 1024 * 1024) == 0:
                            print(f"   {total_bytes // (1024*1024)} MB descargados...")
        
        print(f"✅ Dataset guardado: {output_file} ({total_bytes // (1024*1024)} MB)")
        return output_file
        
    except Exception as e:
        print(f"⚠️ No se pudo descargar dataset: {e}")
        print("   Usando código de ejemplo...")
        
        output_file = output_dir / "code_corpus.txt"
        sample_code = create_sample_code()
        
        # Repetir para tener más datos
        with open(output_file, 'w', encoding='utf-8') as f:
            for _ in range(100):
                f.write(sample_code)
                f.write('\n\n')
        
        print(f"✅ Corpus de ejemplo creado: {output_file}")
        return output_file


def train_tokenizer_spm(corpus_file: Path, output_dir: Path, vocab_size: int = 8000):
    """Entrena tokenizer con SentencePiece."""
    print(f"\n🔧 Entrenando tokenizer SentencePiece (vocab={vocab_size})...")
    
    model_prefix = output_dir / "code_tokenizer"
    
    # Preparar user_defined_symbols (keywords y operadores) - eliminar duplicados
    user_symbols = set(CODE_KEYWORDS) | set(CODE_OPERATORS)
    # Remover los que ya son keywords de Python usados como operadores
    user_symbols.discard('and')
    user_symbols.discard('or')
    user_symbols.discard('not')
    user_symbols_list = list(user_symbols)
    
    spm.SentencePieceTrainer.train(
        input=str(corpus_file),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9995,
        num_threads=8,
        # Tokens especiales
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # User defined symbols (keywords, operadores)
        user_defined_symbols=user_symbols_list,
        # Control de tokens
        split_by_whitespace=True,
        split_by_number=True,
        split_digits=True,
    )
    
    print(f"✅ Tokenizer guardado: {model_prefix}.model")
    return f"{model_prefix}.model"


def train_tokenizer_hf(corpus_file: Path, output_dir: Path, vocab_size: int = 8000):
    """Entrena tokenizer con HuggingFace tokenizers."""
    print(f"\n🔧 Entrenando tokenizer HuggingFace (vocab={vocab_size})...")
    
    # Crear tokenizer BPE
    tokenizer = Tokenizer(models.BPE())
    
    # Pre-tokenizer que preserva estructura de código
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation(),
    ])
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )
    
    # Entrenar
    tokenizer.train([str(corpus_file)], trainer)
    
    # Guardar
    output_file = output_dir / "code_tokenizer.json"
    tokenizer.save(str(output_file))
    
    print(f"✅ Tokenizer guardado: {output_file}")
    return str(output_file)


def test_tokenizer(tokenizer_path: Path):
    """Prueba el tokenizer con código de ejemplo."""
    print("\n🧪 Probando tokenizer...")
    
    if str(tokenizer_path).endswith('.model'):
        sp = spm.SentencePieceProcessor()
        sp.load(str(tokenizer_path))
        
        encode = lambda x: sp.encode(x, out_type=str)
        decode = lambda x: sp.decode(x)
        vocab_size = sp.get_piece_size()
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        encode = lambda x: tokenizer.encode(x).tokens
        decode = lambda x: tokenizer.decode(tokenizer.encode(' '.join(x)).ids)
        vocab_size = tokenizer.get_vocab_size()
    
    print(f"   Vocab size: {vocab_size}")
    
    test_cases = [
        'def fibonacci(n):',
        'if x == 10:',
        'return x + y * 2',
        'async def fetch():',
        'for i in range(10):',
        'class MyClass:',
        '    self.value = 0',
        'fn main() {',
        'let mut x: i32 = 5;',
    ]
    
    print("\n   Ejemplos de tokenización:")
    for code in test_cases:
        tokens = encode(code)
        print(f"   '{code}'")
        print(f"      → {tokens}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Preparar tokenizer para código')
    parser.add_argument('--vocab-size', type=int, default=8000, help='Tamaño del vocabulario')
    parser.add_argument('--corpus-size', type=int, default=50, help='Tamaño del corpus en MB')
    parser.add_argument('--output-dir', type=str, default='data/tokenizer', help='Directorio de salida')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔤 PAMPAr-Coder - Preparar Tokenizer")
    print("=" * 70)
    
    # Crear directorios
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Descargar/crear corpus
    corpus_file = download_code_dataset(output_dir, args.corpus_size)
    
    # 2. Entrenar tokenizer
    if USE_SPM:
        tokenizer_path = train_tokenizer_spm(corpus_file, output_dir, args.vocab_size)
    else:
        tokenizer_path = train_tokenizer_hf(corpus_file, output_dir, args.vocab_size)
    
    # 3. Probar tokenizer
    test_tokenizer(Path(tokenizer_path))
    
    print("\n" + "=" * 70)
    print("✅ Tokenizer listo!")
    print(f"   Archivo: {tokenizer_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
