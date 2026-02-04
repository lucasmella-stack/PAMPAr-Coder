# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Entrenamiento de PAMPAr-Coder con datos de ALTA CALIDAD.

Estrategia:
1. Filtrar datos: solo codigo con docstrings, comentarios, explicaciones
2. Curriculum learning: ejemplos faciles primero, complejos despues
3. Background training: prioridad baja para no molestar
4. Checkpoints frecuentes: resume automatico si se corta
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.coder import PampaRCoder, crear_modelo, CODER_4GB, CODER_4GB_MAX


# =============================================================================
# FILTROS DE CALIDAD
# =============================================================================

def tiene_docstring(code: str) -> bool:
    """Verifica si el codigo tiene docstrings."""
    return '"""' in code or "'''" in code

def tiene_comentarios(code: str) -> bool:
    """Verifica si el codigo tiene comentarios explicativos."""
    lines = code.split('\n')
    comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
    return comment_lines >= 2

def tiene_explicacion(text: str) -> bool:
    """Verifica si hay explicacion en lenguaje natural."""
    keywords = ['porque', 'porque', 'this', 'the', 'we', 'you', 'explicacion',
                'explanation', 'note:', 'nota:', 'importante', 'important',
                'aqui', 'here', 'first', 'primero', 'then', 'luego']
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)

def complejidad_codigo(code: str) -> int:
    """Estima complejidad del codigo (para curriculum learning)."""
    score = 0
    # Longitud
    lines = len(code.split('\n'))
    score += min(lines // 10, 5)
    
    # Estructuras de control
    score += code.count('if ') + code.count('for ') + code.count('while ')
    score += code.count('try:') * 2
    score += code.count('class ') * 3
    score += code.count('async ') * 2
    score += code.count('lambda ') * 2
    
    # Complejidad sintactica
    score += code.count('[') + code.count('{')  # Comprehensions, dicts
    
    return min(score, 20)  # Cap at 20

def es_alta_calidad(item: Dict) -> Tuple[bool, int]:
    """
    Determina si un ejemplo es de alta calidad.
    Returns: (es_calidad, score_complejidad)
    """
    # Extraer texto
    if 'code' in item:
        code = item['code']
        text = item.get('instruction', '') + ' ' + item.get('output', '')
    elif 'content' in item:
        code = item['content']
        text = item.get('prompt', '') + ' ' + code
    elif 'output' in item:
        code = item['output']
        text = item.get('instruction', '') + ' ' + code
    else:
        return False, 0
    
    # Filtros de calidad
    calidad = 0
    
    if tiene_docstring(code):
        calidad += 3
    if tiene_comentarios(code):
        calidad += 2
    if tiene_explicacion(text):
        calidad += 2
    
    # Longitud minima (no snippets muy cortos)
    if len(code) > 100:
        calidad += 1
    if len(code) > 300:
        calidad += 1
    
    # Complejidad
    complejidad = complejidad_codigo(code)
    
    # Umbral de calidad: minimo 3 puntos
    es_calidad = calidad >= 3
    
    return es_calidad, complejidad


# =============================================================================
# DATASET DE ALTA CALIDAD
# =============================================================================

class HighQualityCodeDataset(Dataset):
    """Dataset filtrado por calidad con curriculum learning."""
    
    def __init__(
        self,
        data_dir: Path,
        tokenizer,
        max_length: int = 512,
        max_samples: Optional[int] = None,
        quality_threshold: int = 3,
        curriculum_phase: int = 0  # 0=facil, 1=medio, 2=dificil, 3=todo
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        print(f"\n{'='*60}")
        print(f"Cargando datos de ALTA CALIDAD")
        print(f"{'='*60}")
        
        # Cargar todos los archivos
        all_items = []
        data_path = Path(data_dir)
        
        for jsonl_file in data_path.rglob("*.jsonl"):
            print(f"  Leyendo {jsonl_file.name}...")
            count = 0
            try:
                with open(jsonl_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.strip():
                            try:
                                item = json.loads(line)
                                all_items.append(item)
                                count += 1
                                # Limitar memoria
                                if count >= 500000:
                                    print(f"    Limite alcanzado: {count:,} items")
                                    break
                            except json.JSONDecodeError:
                                continue
                print(f"    Cargados: {count:,}")
            except Exception as e:
                print(f"    Error: {e}")
        
        print(f"\n  Total items cargados: {len(all_items):,}")
        
        # Filtrar por calidad
        print(f"\n  Filtrando por calidad (threshold={quality_threshold})...")
        quality_items = []
        
        for item in all_items:
            es_calidad, complejidad = es_alta_calidad(item)
            if es_calidad:
                quality_items.append((item, complejidad))
        
        print(f"  Items de alta calidad: {len(quality_items):,} ({100*len(quality_items)/len(all_items):.1f}%)")
        
        # Ordenar por complejidad (curriculum learning)
        quality_items.sort(key=lambda x: x[1])
        
        # Dividir en fases
        n = len(quality_items)
        if curriculum_phase == 0:  # Facil
            selected = quality_items[:n//3]
            print(f"  Fase FACIL: {len(selected):,} items")
        elif curriculum_phase == 1:  # Medio
            selected = quality_items[n//3:2*n//3]
            print(f"  Fase MEDIO: {len(selected):,} items")
        elif curriculum_phase == 2:  # Dificil
            selected = quality_items[2*n//3:]
            print(f"  Fase DIFICIL: {len(selected):,} items")
        else:  # Todo
            selected = quality_items
            print(f"  Fase COMPLETA: {len(selected):,} items")
        
        # Limitar si es necesario
        if max_samples and len(selected) > max_samples:
            selected = selected[:max_samples]
            print(f"  Limitado a: {len(selected):,} items")
        
        # Preparar samples
        print(f"\n  Tokenizando...")
        for item, _ in selected:
            text = self._extract_text(item)
            if text and len(text) > 50:
                self.samples.append(text)
        
        print(f"  Samples finales: {len(self.samples):,}")
        print(f"{'='*60}\n")
    
    def _extract_text(self, item: Dict) -> str:
        """Extrae texto del item."""
        if 'code' in item:
            instruction = item.get('instruction', '')
            code = item['code']
            output = item.get('output', code)
            return f"# Tarea: {instruction}\n\n{output}" if instruction else output
        elif 'content' in item:
            return item['content']
        elif 'output' in item:
            instruction = item.get('instruction', '')
            output = item['output']
            return f"# Tarea: {instruction}\n\n{output}" if instruction else output
        return ""
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        
        # Tokenizar
        tokens = self.tokenizer.encode(text)
        
        # Truncar/pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Input y target (shift by 1)
        input_ids = tokens[:-1]
        targets = tokens[1:]
        
        return input_ids, targets


# =============================================================================
# TRAINER CON CURRICULUM LEARNING
# =============================================================================

class QualityTrainer:
    """Trainer optimizado para datos de alta calidad."""
    
    def __init__(
        self,
        model: PampaRCoder,
        tokenizer,
        data_dir: Path,
        checkpoint_dir: Path,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        use_amp: bool = True,
        low_priority: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.batch_size = batch_size
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # AMP (con sintaxis nueva para evitar deprecation warnings)
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Estado
        self.current_phase = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Prioridad baja (Windows)
        if low_priority and sys.platform == 'win32':
            import psutil
            p = psutil.Process()
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            print("  Prioridad del proceso: BAJA (no molestara tu trabajo)")
    
    def train_phase(
        self,
        phase: int,
        epochs: int,
        max_samples: Optional[int] = None
    ):
        """Entrena una fase del curriculum."""
        phase_names = ['FACIL', 'MEDIO', 'DIFICIL', 'COMPLETO']
        print(f"\n{'='*70}")
        print(f"  FASE {phase}: {phase_names[phase]}")
        print(f"{'='*70}")
        
        # Dataset para esta fase
        dataset = HighQualityCodeDataset(
            data_dir=self.data_dir,
            tokenizer=self.tokenizer,
            max_length=self.model.config.max_seq_len,
            max_samples=max_samples,
            curriculum_phase=phase
        )
        
        if len(dataset) == 0:
            print("  No hay datos para esta fase, saltando...")
            return
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        # LR scheduler
        total_steps = len(dataloader) * epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
        
        # Training loop
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_samples = 0
            start_time = time.time()
            
            for batch_idx, (input_ids, targets) in enumerate(dataloader):
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                
                # Forward
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits, loss = self.model(input_ids, targets)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits, loss = self.model(input_ids, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                scheduler.step()
                
                # Stats
                epoch_loss += loss.item()
                epoch_samples += input_ids.size(0)
                self.global_step += 1
                
                # Log cada 50 batches
                if batch_idx % 50 == 0:
                    elapsed = time.time() - start_time
                    samples_per_sec = epoch_samples / elapsed if elapsed > 0 else 0
                    lr = scheduler.get_last_lr()[0]
                    print(f"    Batch {batch_idx:>5}/{len(dataloader)} | "
                          f"Loss: {loss.item():.4f} | "
                          f"LR: {lr:.2e} | "
                          f"{samples_per_sec:.1f} samples/s")
            
            # Epoch stats
            avg_loss = epoch_loss / len(dataloader)
            elapsed = time.time() - start_time
            
            print(f"\n  Epoca {epoch+1}/{epochs} completada:")
            print(f"    Loss promedio: {avg_loss:.4f}")
            print(f"    Perplexity: {torch.exp(torch.tensor(avg_loss)):.2f}")
            print(f"    Tiempo: {elapsed/60:.1f} min")
            
            # Checkpoint
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(f"best_phase{phase}.pt")
                print(f"    Mejor modelo guardado!")
            
            self.save_checkpoint("last.pt")
    
    def save_checkpoint(self, filename: str):
        """Guarda checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.model.config.__dict__,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'current_phase': self.current_phase,
        }, path)
    
    def load_checkpoint(self, filename: str) -> bool:
        """Carga checkpoint si existe."""
        path = self.checkpoint_dir / filename
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.global_step = checkpoint.get('global_step', 0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.current_phase = checkpoint.get('current_phase', 0)
            print(f"  Checkpoint cargado: {filename}")
            print(f"    Global step: {self.global_step}")
            print(f"    Best loss: {self.best_loss:.4f}")
            return True
        return False
    
    def train_full_curriculum(
        self,
        epochs_per_phase: int = 3,
        max_samples_per_phase: Optional[int] = None
    ):
        """Entrenamiento completo con curriculum learning."""
        print("\n" + "="*70)
        print("  ENTRENAMIENTO CON CURRICULUM LEARNING")
        print("  Fases: FACIL -> MEDIO -> DIFICIL -> COMPLETO")
        print("="*70)
        
        # Intentar resume
        self.load_checkpoint("last.pt")
        
        # Entrenar cada fase
        for phase in range(4):
            if phase < self.current_phase:
                print(f"\n  Saltando fase {phase} (ya completada)")
                continue
            
            self.current_phase = phase
            self.train_phase(
                phase=phase,
                epochs=epochs_per_phase,
                max_samples=max_samples_per_phase
            )
        
        print("\n" + "="*70)
        print("  ENTRENAMIENTO COMPLETADO!")
        print(f"  Mejor loss: {self.best_loss:.4f}")
        print("="*70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train PAMPAr-Coder with high-quality data')
    parser.add_argument('--epochs', type=int, default=3, help='Epochs per phase')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1.5e-4, help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples per phase')
    parser.add_argument('--preset', type=str, default='4GB', choices=['4GB', '4GB_MAX'])
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--phase', type=int, default=None, help='Train specific phase only')
    args = parser.parse_args()
    
    print("="*70)
    print("  PAMPAr-Coder - Entrenamiento de ALTA CALIDAD")
    print("="*70)
    
    # Modelo
    print("\n  Creando modelo...")
    if args.preset == '4GB_MAX':
        model = crear_modelo("4GB_MAX")
    else:
        model = crear_modelo("4GB")
    
    params = model.count_parameters()
    print(f"    Parametros: {params['total']:,}")
    
    # Tokenizer
    print("\n  Cargando tokenizer...")
    import sentencepiece as spm
    tokenizer = spm.SentencePieceProcessor()
    tokenizer_path = Path("data/tokenizer/code_tokenizer.model")
    if tokenizer_path.exists():
        tokenizer.load(str(tokenizer_path))
        print(f"    Vocab size: {tokenizer.get_piece_size()}")
        
        # Ajustar vocab si es necesario
        if tokenizer.get_piece_size() != model.config.vocab_size:
            print(f"    Ajustando vocab_size: {model.config.vocab_size} -> {tokenizer.get_piece_size()}")
            model.config.vocab_size = tokenizer.get_piece_size()
            # Recrear modelo con vocab correcto
            model = PampaRCoder(model.config)
    else:
        print("    ERROR: Tokenizer no encontrado!")
        return
    
    # Device info
    if torch.cuda.is_available():
        print(f"\n  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Trainer
    trainer = QualityTrainer(
        model=model,
        tokenizer=tokenizer,
        data_dir=Path("data"),
        checkpoint_dir=Path("checkpoints"),
        learning_rate=args.lr,
        batch_size=args.batch_size,
        use_amp=True,
        low_priority=True
    )
    
    # Entrenar
    if args.phase is not None:
        trainer.train_phase(
            phase=args.phase,
            epochs=args.epochs,
            max_samples=args.max_samples
        )
    else:
        trainer.train_full_curriculum(
            epochs_per_phase=args.epochs,
            max_samples_per_phase=args.max_samples
        )


if __name__ == "__main__":
    main()
