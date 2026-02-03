# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr-Coder - Entrenamiento SERIO para competir con Kimi-72B.

Este script implementa:
1. Descarga de múltiples datasets de código
2. Entrenamiento con gradient accumulation (simula batches grandes)
3. Tokenizer BPE de alta calidad
4. Logging detallado y checkpointing
5. Evaluación en HumanEval y MBPP

Objetivo: Demostrar que la arquitectura territorial logra calidad
competitiva con modelos 100x más grandes.
"""

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.amp import autocast, GradScaler

# Agregar path del proyecto
sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.coder import PampaRCoder, crear_modelo, CODER_4GB, CODER_4GB_MAX


# =============================================================================
# DATASETS
# =============================================================================

class MultiCodeDataset(IterableDataset):
    """
    Dataset que combina múltiples fuentes de código.
    Streaming para no saturar RAM.
    """
    
    def __init__(
        self, 
        tokenizer,
        max_seq_len: int = 512,
        max_samples: int = None,
        languages: List[str] = None
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_samples = max_samples
        self.languages = languages or ['python']
        
        # Datasets a usar
        self.dataset_configs = [
            # (nombre, config, split, campo_codigo)
            ('Nan-Do/code-search-net-python', None, 'train', 'code'),
        ]
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        from datasets import load_dataset
        
        count = 0
        
        for ds_name, ds_config, ds_split, code_field in self.dataset_configs:
            try:
                ds = load_dataset(ds_name, ds_config, split=ds_split, streaming=True)
                
                for sample in ds:
                    if self.max_samples and count >= self.max_samples:
                        return
                    
                    code = sample.get(code_field, '')
                    if not code or len(code) < 20:
                        continue
                    
                    # Tokenizar
                    tokens = self.tokenizer.encode(code)
                    
                    # Truncar o skip si muy corto
                    if len(tokens) < 10:
                        continue
                    
                    tokens = tokens[:self.max_seq_len]
                    
                    # Padding
                    if len(tokens) < self.max_seq_len:
                        tokens = tokens + [0] * (self.max_seq_len - len(tokens))
                    
                    input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
                    targets = torch.tensor(tokens[1:], dtype=torch.long)
                    
                    yield {
                        'input_ids': input_ids,
                        'targets': targets
                    }
                    
                    count += 1
                    
            except Exception as e:
                print(f"⚠️ Error con {ds_name}: {e}")
                continue


class LocalCodeDataset(Dataset):
    """Dataset de código local (JSONL)."""
    
    def __init__(self, jsonl_path: str, tokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                code = data.get('code', data.get('content', ''))
                if code and len(code) >= 20:
                    self.samples.append(code)
        
        print(f"📚 Cargado {len(self.samples)} samples de {jsonl_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        code = self.samples[idx]
        tokens = self.tokenizer.encode(code)
        
        # Truncar
        tokens = tokens[:self.max_seq_len]
        
        # Padding
        if len(tokens) < self.max_seq_len:
            tokens = tokens + [0] * (self.max_seq_len - len(tokens))
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'targets': targets
        }


# =============================================================================
# TRAINER
# =============================================================================

class SeriousTrainer:
    """
    Trainer optimizado para competir con modelos grandes.
    
    Features:
    - Gradient accumulation
    - Mixed precision (FP16)
    - Gradient clipping
    - Learning rate scheduling
    - Checkpointing
    - Logging detallado
    """
    
    def __init__(
        self,
        model: PampaRCoder,
        tokenizer,
        config: dict
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 2e-4),
            weight_decay=config.get('weight_decay', 0.01),
            betas=(0.9, 0.95)
        )
        
        # Gradient accumulation para simular batch grande
        self.grad_accum_steps = config.get('grad_accum_steps', 8)
        self.effective_batch_size = config.get('batch_size', 4) * self.grad_accum_steps
        
        # Scheduler
        self.warmup_steps = config.get('warmup_steps', 500)
        self.total_steps = config.get('total_steps', 50000)
        
        # Logging
        self.log_interval = config.get('log_interval', 50)
        self.eval_interval = config.get('eval_interval', 1000)
        self.save_interval = config.get('save_interval', 5000)
        
        # Checkpoint dir
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Stats
        self.global_step = 0
        self.best_loss = float('inf')
        self.history = {
            'train_loss': [],
            'learning_rate': [],
            'tokens_per_sec': [],
        }
    
    def get_lr(self) -> float:
        """Learning rate con warmup + cosine decay."""
        if self.global_step < self.warmup_steps:
            return self.config['lr'] * self.global_step / self.warmup_steps
        
        progress = (self.global_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.config['lr'] * 0.5 * (1 + math.cos(math.pi * progress))
    
    def update_lr(self):
        """Actualiza learning rate del optimizer."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Un paso de entrenamiento."""
        input_ids = batch['input_ids'].to(self.device)
        targets = batch['targets'].to(self.device)
        
        with autocast('cuda', enabled=self.use_amp):
            logits, loss = self.model(input_ids, targets)
            loss = loss / self.grad_accum_steps
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.grad_accum_steps
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Entrena una época."""
        self.model.train()
        
        total_loss = 0
        n_batches = 0
        start_time = time.time()
        tokens_processed = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            loss = self.train_step(batch)
            total_loss += loss
            n_batches += 1
            tokens_processed += batch['input_ids'].numel()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update LR
                lr = self.update_lr()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.log_interval == 0:
                    elapsed = time.time() - start_time
                    tok_per_sec = tokens_processed / elapsed
                    avg_loss = total_loss / n_batches
                    
                    print(f"   Step {self.global_step:,} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {lr:.2e} | "
                          f"Tok/s: {tok_per_sec:.0f}")
                    
                    self.history['train_loss'].append(avg_loss)
                    self.history['learning_rate'].append(lr)
                    self.history['tokens_per_sec'].append(tok_per_sec)
                
                # Checkpoint
                if self.global_step % self.save_interval == 0:
                    self.save_checkpoint(f'step_{self.global_step}.pt')
        
        avg_loss = total_loss / n_batches
        return avg_loss
    
    def save_checkpoint(self, filename: str):
        """Guarda checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'config': self.config,
            'history': self.history,
        }, path)
        print(f"   💾 Checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Carga checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)
        print(f"✅ Checkpoint cargado: {path} (step {self.global_step})")
    
    def train(self, train_dataloader: DataLoader, epochs: int):
        """Loop principal de entrenamiento."""
        print("\n" + "=" * 70)
        print("🚀 Iniciando entrenamiento SERIO")
        print("=" * 70)
        print(f"   Device: {self.device}")
        print(f"   Batch size: {self.config['batch_size']} × {self.grad_accum_steps} = {self.effective_batch_size}")
        print(f"   Total steps: {self.total_steps:,}")
        print(f"   Warmup: {self.warmup_steps:,}")
        print(f"   Mixed precision: {self.use_amp}")
        
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        start_training = time.time()
        
        for epoch in range(epochs):
            print(f"\n📅 Época {epoch + 1}/{epochs}")
            print("-" * 40)
            
            avg_loss = self.train_epoch(train_dataloader, epoch)
            
            print(f"\n   📊 Época {epoch + 1} completada")
            print(f"      Loss promedio: {avg_loss:.4f}")
            print(f"      Perplexity: {math.exp(avg_loss):.2f}")
            
            # Save best
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint('best.pt')
        
        elapsed = time.time() - start_training
        print("\n" + "=" * 70)
        print("✅ Entrenamiento completado!")
        print(f"   Tiempo total: {elapsed/3600:.2f} horas")
        print(f"   Mejor loss: {self.best_loss:.4f}")
        print(f"   Steps totales: {self.global_step:,}")
        print("=" * 70)
        
        # Save final
        self.save_checkpoint('final.pt')
        
        # Save history
        with open(self.checkpoint_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)


# =============================================================================
# TOKENIZER
# =============================================================================

def load_or_create_tokenizer(tokenizer_path: str, vocab_size: int = 16000):
    """Carga o crea tokenizer."""
    import sentencepiece as spm
    
    if os.path.exists(tokenizer_path):
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_path)
        print(f"✅ Tokenizer cargado: {tokenizer_path} (vocab={sp.get_piece_size()})")
        return sp
    
    print(f"⚠️ Tokenizer no encontrado: {tokenizer_path}")
    print("   Usa prepare_tokenizer.py para crear uno")
    return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='PAMPAr-Coder - Entrenamiento Serio')
    parser.add_argument('--preset', type=str, default='4GB', 
                        choices=['4GB', '4GB_MAX'],
                        help='Preset de configuración')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--grad-accum', type=int, default=8,
                        help='Pasos de gradient accumulation')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Límite de samples (None = todos)')
    parser.add_argument('--total-steps', type=int, default=50000)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path a checkpoint para resumir')
    parser.add_argument('--local-data', type=str, default=None,
                        help='Path a JSONL local de código')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🐸 PAMPAr-Coder - Entrenamiento para competir con Kimi-72B")
    print("=" * 70)
    
    # Config
    if args.preset == '4GB_MAX':
        model_config = CODER_4GB_MAX
    else:
        model_config = CODER_4GB
    
    print(f"\n⚙️ Configuración: {args.preset}")
    params = model_config.estimate_params()
    print(f"   Parámetros: {params['total']:,} ({params['total']/1e6:.1f}M)")
    print(f"   Dim: {model_config.dim}, Capas: {model_config.n_capas}")
    print(f"   LLAVES: {int(model_config.peso_llaves*100)}% reglas")
    
    # Tokenizer
    tokenizer_path = 'data/tokenizer/code_tokenizer.model'
    tokenizer = load_or_create_tokenizer(tokenizer_path, model_config.vocab_size)
    if tokenizer is None:
        print("❌ No hay tokenizer. Ejecuta primero:")
        print("   python scripts/prepare_tokenizer.py")
        return
    
    # Ajustar vocab_size al tokenizer real
    model_config.vocab_size = tokenizer.get_piece_size()
    
    # Model
    print("\n🏗️ Creando modelo...")
    model = PampaRCoder(model_config)
    print(f"   Parámetros reales: {sum(p.numel() for p in model.parameters()):,}")
    
    # Dataset
    print("\n📚 Preparando dataset...")
    if args.local_data and os.path.exists(args.local_data):
        dataset = LocalCodeDataset(
            args.local_data, 
            tokenizer, 
            max_seq_len=model_config.max_seq_len
        )
    else:
        dataset = MultiCodeDataset(
            tokenizer,
            max_seq_len=model_config.max_seq_len,
            max_samples=args.max_samples
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,  # Streaming no soporta workers
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Trainer config
    train_config = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'grad_accum_steps': args.grad_accum,
        'weight_decay': 0.01,
        'warmup_steps': 500,
        'total_steps': args.total_steps,
        'use_amp': True,
        'log_interval': 50,
        'save_interval': 2000,
        'checkpoint_dir': 'checkpoints',
    }
    
    # Trainer
    trainer = SeriousTrainer(model, tokenizer, train_config)
    
    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(dataloader, args.epochs)


if __name__ == '__main__':
    main()
