# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Script de entrenamiento para PAMPAr-Coder.

Entrena el modelo territorial especializado en código.
Optimizado para hardware consumer (GTX 1650, 4GB VRAM).
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Agregar path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.coder import PampaRCoder, crear_modelo, CODER_4GB, CODER_8GB, CODER_24GB

try:
    import sentencepiece as spm
    HAS_SPM = True
except ImportError:
    HAS_SPM = False


# =============================================================================
# Dataset
# =============================================================================

class CodeDataset(Dataset):
    """Dataset de código para entrenamiento."""
    
    def __init__(
        self,
        data_file: Path,
        tokenizer_path: Path,
        max_seq_len: int = 512,
        max_samples: Optional[int] = None
    ):
        self.max_seq_len = max_seq_len
        
        # Cargar tokenizer
        if str(tokenizer_path).endswith('.model'):
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(str(tokenizer_path))
            self.encode = lambda x: self.tokenizer.encode(x)
            self.vocab_size = self.tokenizer.get_piece_size()
            self.pad_id = self.tokenizer.pad_id()
            self.bos_id = self.tokenizer.bos_id()
            self.eos_id = self.tokenizer.eos_id()
        else:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self.encode = lambda x: self.tokenizer.encode(x).ids
            self.vocab_size = self.tokenizer.get_vocab_size()
            self.pad_id = 0
            self.bos_id = 2
            self.eos_id = 3
        
        # Cargar datos
        self.samples = []
        print(f"📚 Cargando {data_file}...")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                
                try:
                    record = json.loads(line)
                    text = record.get('text', '')
                    if len(text) > 50:
                        self.samples.append(text)
                except:
                    continue
        
        print(f"   {len(self.samples):,} samples cargados")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        
        # Tokenizar
        tokens = self.encode(text)
        
        # Truncar/padding
        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[:self.max_seq_len - 2]
        
        # Agregar BOS/EOS
        tokens = [self.bos_id] + tokens + [self.eos_id]
        
        # Padding
        padding = [self.pad_id] * (self.max_seq_len - len(tokens))
        tokens = tokens + padding
        
        # Input y target (shifted)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        # Mask para padding (-100 ignora en loss)
        target_ids[target_ids == self.pad_id] = -100
        
        return input_ids, target_ids


# =============================================================================
# Training Loop
# =============================================================================

class Trainer:
    """Entrenador para PAMPAr-Coder."""
    
    def __init__(
        self,
        model: PampaRCoder,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        config: dict
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"🖥️ Device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,  # Windows compatibility
            pin_memory=True if self.device.type == 'cuda' else False,
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'] * 2,
                shuffle=False,
                num_workers=0,
            )
        else:
            self.val_loader = None
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.95),
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * config['epochs']
        warmup_steps = config.get('warmup_steps', 100)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, epoch: int) -> float:
        """Entrena una época."""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        start_time = time.time()
        
        for batch_idx, (input_ids, targets) in enumerate(self.train_loader):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
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
            
            self.scheduler.step()
            
            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1
            
            # Logging
            if batch_idx % 50 == 0:
                elapsed = time.time() - start_time
                samples_per_sec = (batch_idx + 1) * self.config['batch_size'] / elapsed
                lr = self.scheduler.get_last_lr()[0]
                
                print(f"   Batch {batch_idx:4d}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {lr:.2e} | "
                      f"{samples_per_sec:.1f} samples/s")
        
        avg_loss = total_loss / n_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """Valida el modelo."""
        if not self.val_loader:
            return float('inf')
        
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        for input_ids, targets in self.val_loader:
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            if self.use_amp:
                with autocast():
                    _, loss = self.model(input_ids, targets)
            else:
                _, loss = self.model(input_ids, targets)
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Guarda checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        # Guardar último
        path = self.checkpoint_dir / "last.pt"
        torch.save(checkpoint, path)
        
        # Guardar mejor
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"   💾 Mejor modelo guardado: {best_path}")
    
    def train(self):
        """Loop de entrenamiento completo."""
        print("\n" + "=" * 70)
        print("🚀 Iniciando entrenamiento")
        print("=" * 70)
        
        epochs = self.config['epochs']
        
        for epoch in range(epochs):
            print(f"\n📅 Época {epoch + 1}/{epochs}")
            print("-" * 40)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Check best
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Report
            print(f"\n   📊 Resumen época {epoch + 1}:")
            print(f"      Train loss: {train_loss:.4f}")
            print(f"      Val loss:   {val_loss:.4f}")
            print(f"      Best val:   {self.best_val_loss:.4f}")
            
            # Perplexity
            train_ppl = math.exp(train_loss) if train_loss < 100 else float('inf')
            val_ppl = math.exp(val_loss) if val_loss < 100 else float('inf')
            print(f"      Train PPL:  {train_ppl:.2f}")
            print(f"      Val PPL:    {val_ppl:.2f}")
        
        print("\n" + "=" * 70)
        print("✅ Entrenamiento completado!")
        print(f"   Mejor val loss: {self.best_val_loss:.4f}")
        print(f"   Checkpoints en: {self.checkpoint_dir}")
        print("=" * 70)
        
        return self.train_losses, self.val_losses


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Entrenar PAMPAr-Coder')
    
    # Data
    parser.add_argument('--train-data', type=str, default='data/code/train.jsonl')
    parser.add_argument('--val-data', type=str, default='data/code/val.jsonl')
    parser.add_argument('--tokenizer', type=str, default='data/tokenizer/code_tokenizer.model')
    
    # Model
    parser.add_argument('--preset', type=str, default='4GB', 
                        choices=['4GB', '8GB', '24GB'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from checkpoint')
    
    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--max-samples', type=int, default=None)
    
    # Output
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🧠 PAMPAr-Coder - Entrenamiento")
    print("=" * 70)
    
    # Configuración
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'warmup_steps': args.warmup_steps,
        'checkpoint_dir': args.checkpoint_dir,
        'use_amp': True,
    }
    
    print(f"\n⚙️ Configuración:")
    for k, v in config.items():
        print(f"   {k}: {v}")
    
    # Crear modelo
    print(f"\n🏗️ Creando modelo (preset={args.preset})...")
    model = crear_modelo(args.preset)
    
    params = model.count_parameters()
    print(f"   Parámetros: {params['total']:,}")
    
    # Actualizar vocab_size si hay tokenizer
    tokenizer_path = Path(args.tokenizer)
    if tokenizer_path.exists():
        if str(tokenizer_path).endswith('.model'):
            sp = spm.SentencePieceProcessor()
            sp.load(str(tokenizer_path))
            vocab_size = sp.get_piece_size()
        else:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(tokenizer_path))
            vocab_size = tok.get_vocab_size()
        
        if vocab_size != model.config.vocab_size:
            print(f"   ⚠️ Ajustando vocab_size: {model.config.vocab_size} → {vocab_size}")
            model.config.vocab_size = vocab_size
            # Recrear modelo con vocab correcto
            model = PampaRCoder(model.config)
    
    # Cargar checkpoint si existe
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"\n📂 Cargando checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    
    # Cargar datasets
    print(f"\n📚 Cargando datasets...")
    
    train_dataset = CodeDataset(
        Path(args.train_data),
        tokenizer_path,
        max_seq_len=model.config.max_seq_len,
        max_samples=args.max_samples
    )
    
    val_path = Path(args.val_data)
    val_dataset = CodeDataset(
        val_path,
        tokenizer_path,
        max_seq_len=model.config.max_seq_len,
        max_samples=args.max_samples // 10 if args.max_samples else None
    ) if val_path.exists() else None
    
    # Entrenar
    trainer = Trainer(model, train_dataset, val_dataset, config)
    train_losses, val_losses = trainer.train()
    
    # Guardar historial
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config,
    }
    history_path = Path(args.checkpoint_dir) / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n📊 Historial guardado: {history_path}")


if __name__ == "__main__":
    main()
