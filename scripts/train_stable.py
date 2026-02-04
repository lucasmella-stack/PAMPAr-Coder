# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Entrenamiento ESTABLE de PAMPAr-Coder.

Version simplificada para evitar problemas de memoria.
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.coder import PampaRCoder, crear_modelo, CODER_4GB


def limpiar_memoria():
    """Limpia memoria GPU y RAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class QualityFilter:
    """Filtro de calidad para código."""
    
    @staticmethod
    def has_docstring(code: str) -> bool:
        return '"""' in code or "'''" in code
    
    @staticmethod
    def has_comments(code: str) -> bool:
        return '#' in code
    
    @staticmethod
    def has_explanation(text: str) -> bool:
        markers = ['because', 'since', 'therefore', 'note:', 'explanation:',
                   'this code', 'this function', 'we need', 'the purpose',
                   'porque', 'ya que', 'esto', 'esta función', 'necesitamos']
        text_lower = text.lower()
        return any(m in text_lower for m in markers)
    
    @staticmethod
    def score_quality(item: Dict) -> int:
        score = 0
        text = str(item.get('instruction', '')) + ' ' + str(item.get('response', ''))
        text += str(item.get('input', '')) + ' ' + str(item.get('output', ''))
        text += str(item.get('text', '')) + ' ' + str(item.get('code', ''))
        
        if QualityFilter.has_docstring(text):
            score += 2
        if QualityFilter.has_comments(text):
            score += 1
        if QualityFilter.has_explanation(text):
            score += 2
        if len(text) > 200:
            score += 1
        
        return score


class SimpleDataset(Dataset):
    """Dataset simple para código."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        
        # Truncar/Pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens[:-1], tokens[1:]


def load_quality_data(data_dir: str, quality_threshold: int = 3, max_items: int = 20000) -> List[str]:
    """Carga datos filtrados por calidad."""
    
    data_path = Path(data_dir)
    items = []
    
    # Cargar todos los JSONL
    for jsonl_file in data_path.rglob("*.jsonl"):
        print(f"  Leyendo {jsonl_file.name}...")
        try:
            with open(jsonl_file, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= 100000:  # Limite por archivo
                        break
                    try:
                        item = json.loads(line.strip())
                        items.append(item)
                    except:
                        continue
        except Exception as e:
            print(f"    Error: {e}")
    
    print(f"  Total cargados: {len(items)}")
    
    # Filtrar por calidad
    print(f"  Filtrando por calidad...")
    quality_items = []
    for item in items:
        if QualityFilter.score_quality(item) >= quality_threshold:
            quality_items.append(item)
            if len(quality_items) >= max_items:
                break
    
    print(f"  Items de alta calidad: {len(quality_items)}")
    
    # Convertir a textos
    texts = []
    for item in quality_items:
        text = ""
        if 'instruction' in item:
            text = f"### Instrucción:\n{item['instruction']}\n\n### Respuesta:\n{item.get('response', item.get('output', ''))}"
        elif 'text' in item:
            text = item['text']
        elif 'code' in item:
            text = item['code']
        
        if len(text) > 50:
            texts.append(text)
    
    return texts


def train_epoch(model, dataloader, optimizer, device, epoch_num):
    """Entrena una época."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward (sin AMP para estabilidad)
        logits, loss = model(input_ids, targets)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Limpiar memoria cada 10 batches
        if batch_idx % 10 == 0:
            limpiar_memoria()
        
        # Log cada 50 batches
        if batch_idx % 50 == 0:
            avg_loss = total_loss / num_batches
            print(f"    Batch {batch_idx:>5}/{len(dataloader)} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Entrenamiento estable PAMPAr-Coder')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-samples', type=int, default=10000)
    args = parser.parse_args()
    
    print("=" * 60)
    print("  PAMPAr-Coder - Entrenamiento ESTABLE")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Crear modelo
    print("\n  Creando modelo...")
    model = crear_modelo("4GB")  # Usa el preset CODER_4GB
    model = model.to(device)
    config_dict = {'preset': '4GB'}  # Para guardar en checkpoint
    print(f"    Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Tokenizer
    print("\n  Cargando tokenizer...")
    import sentencepiece as spm
    tokenizer_path = Path(__file__).parent.parent / "data" / "tokenizer" / "code_tokenizer.model"
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    model.registrar_tokenizer(tokenizer)
    print(f"    Vocab size: {tokenizer.vocab_size()}")
    
    # Cargar datos
    print("\n  Cargando datos de calidad...")
    data_dir = Path(__file__).parent.parent / "data"
    texts = load_quality_data(str(data_dir), quality_threshold=3, max_items=args.max_samples)
    print(f"    Textos finales: {len(texts)}")
    
    # Dataset y DataLoader
    dataset = SimpleDataset(texts, tokenizer, max_length=256)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Sin workers para estabilidad
        pin_memory=False
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Checkpoint dir
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Cargar checkpoint si existe
    checkpoint_path = checkpoint_dir / "stable_last.pt"
    start_epoch = 0
    if checkpoint_path.exists():
        print(f"\n  Cargando checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        print(f"    Continuando desde época {start_epoch}")
    
    # Entrenar
    print("\n" + "=" * 60)
    print("  ENTRENAMIENTO")
    print("=" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n  Época {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        epoch_start = time.time()
        avg_loss = train_epoch(model, dataloader, optimizer, device, epoch)
        epoch_time = time.time() - epoch_start
        
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        print(f"\n  Época {epoch+1} completada:")
        print(f"    Loss: {avg_loss:.4f}")
        print(f"    Perplexity: {perplexity:.2f}")
        print(f"    Tiempo: {epoch_time/60:.1f} min")
        
        # Guardar checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'config': config_dict,
            'epoch': epoch,
            'loss': avg_loss
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"    Guardado: {checkpoint_path.name}")
        
        # Guardar mejor
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = checkpoint_dir / "stable_best.pt"
            torch.save(checkpoint, best_path)
            print(f"    ¡Nuevo mejor modelo! -> {best_path.name}")
        
        limpiar_memoria()
    
    print("\n" + "=" * 60)
    print("  ¡ENTRENAMIENTO COMPLETADO!")
    print(f"  Mejor loss: {best_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
