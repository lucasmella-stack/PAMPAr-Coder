# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr-Coder: Entrenamiento en Cloud (RunPod A40).

Script robusto con:
- Checkpointing frecuente
- Recuperación de errores
- Logging con wandb
- Early stopping
- Gradient clipping
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
import sentencepiece as spm
from tqdm import tqdm

# Agregar path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pampar.coder.model_v2 import PampaRCoderV2, ConfigPampaRCoderV2
from pampar.coder.zonas_brodmann import MacroTerritorio
from config_3b import Config3B, Config1_5B, CONFIGS


# =============================================================================
# DATASET
# =============================================================================

class StreamingCodeDataset(Dataset):
    """
    Dataset optimizado para grandes volúmenes.
    
    Features:
    - Lazy loading (no carga todo en RAM)
    - Pre-tokenizado para velocidad
    - Shuffling eficiente
    """
    
    def __init__(
        self,
        data_paths: List[str],
        tokenizer_path: str,
        max_length: int = 2048,
        max_samples: Optional[int] = None,
    ):
        self.max_length = max_length
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        
        # Index de archivos y líneas
        self.file_indices: List[Tuple[str, int]] = []
        
        total = 0
        for path in data_paths:
            if not Path(path).exists():
                continue
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_samples and total >= max_samples:
                        break
                    self.file_indices.append((path, i))
                    total += 1
        
        print(f"📚 Dataset indexado: {len(self.file_indices)} samples")
    
    def __len__(self):
        return len(self.file_indices)
    
    def __getitem__(self, idx):
        path, line_idx = self.file_indices[idx]
        
        # Leer línea específica
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == line_idx:
                    data = json.loads(line)
                    break
        
        # Extraer texto
        text = self._extract_text(data)
        
        # Tokenizar
        tokens = self.tokenizer.Encode(text)[:self.max_length]
        
        # Padding
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        
        return {"input_ids": input_ids, "labels": labels}
    
    def _extract_text(self, data: Dict) -> str:
        """Extrae texto de diferentes formatos."""
        if "text" in data:
            return data["text"]
        if "instruction" in data:
            response = data.get("response", "") or data.get("output", "")
            return f"[INST]{data['instruction']}[/INST]{response}"
        if "code" in data:
            return data["code"]
        if "content" in data:
            return data["content"]
        return str(data)


# =============================================================================
# TRAINER
# =============================================================================

class CloudTrainer:
    """
    Trainer robusto para cloud.
    
    Features:
    - Checkpoint recovery automático
    - Wandb logging
    - Early stopping
    - Gradient monitoring
    """
    
    def __init__(
        self,
        model: PampaRCoderV2,
        config: Config3B,
        tokenizer_path: str,
        output_dir: str = "checkpoints",
        use_wandb: bool = True,
        project_name: str = "pampar-coder-3b",
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        model.registrar_tokenizer(self.tokenizer)
        
        # Mixed precision
        self.scaler = GradScaler('cuda') if config.use_mixed_precision else None
        
        # Wandb
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=project_name,
                    config=asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config),
                )
                self.wandb = wandb
            except:
                self.use_wandb = False
                print("⚠️ wandb no disponible, continuando sin logging")
        
        # Métricas
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.no_improve_count = 0
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        patience: int = 5,
        resume_from: Optional[str] = None,
    ):
        """Entrenamiento principal."""
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )
        
        # Scheduler
        total_steps = len(train_loader) * epochs // self.config.gradient_accumulation
        
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            progress = (step - self.config.warmup_steps) / max(1, total_steps - self.config.warmup_steps)
            return 0.1 + 0.9 * (1 - progress)
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Resume
        start_epoch = 0
        if resume_from and Path(resume_from).exists():
            start_epoch = self._load_checkpoint(resume_from, optimizer, scheduler)
            print(f"🔄 Resumido desde epoch {start_epoch}")
        
        # Training loop
        print(f"\n{'='*60}")
        print(f"🚀 Iniciando entrenamiento")
        print(f"   Epochs: {epochs}")
        print(f"   Steps por epoch: {len(train_loader)}")
        print(f"   Total steps: {total_steps}")
        print(f"{'='*60}\n")
        
        for epoch in range(start_epoch, epochs):
            # Train
            train_loss, train_metrics = self._train_epoch(
                train_loader, optimizer, scheduler
            )
            
            # Eval
            val_loss, val_metrics = self._evaluate(val_loader)
            
            # Log
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            if self.use_wandb:
                self.wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    **train_metrics,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                })
            
            # Save checkpoint
            self._save_checkpoint(epoch + 1, optimizer, scheduler, val_loss)
            
            # Best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.no_improve_count = 0
                self._save_checkpoint(
                    epoch + 1, optimizer, scheduler, val_loss,
                    filename="best_model.pt"
                )
                print(f"   ✨ Nuevo mejor modelo!")
            else:
                self.no_improve_count += 1
            
            # Early stopping
            if self.no_improve_count >= patience:
                print(f"⏹️ Early stopping: {patience} epochs sin mejora")
                break
        
        print(f"\n✅ Entrenamiento completado!")
        print(f"   Mejor val_loss: {self.best_val_loss:.4f}")
    
    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ) -> Tuple[float, Dict]:
        """Entrena una epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        grad_norms = []
        
        optimizer.zero_grad()
        pbar = tqdm(dataloader, desc="Training")
        
        for i, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward
            if self.scaler:
                with autocast('cuda'):
                    logits, loss, _ = self.model(input_ids, labels)
                self.scaler.scale(loss / self.config.gradient_accumulation).backward()
            else:
                logits, loss, _ = self.model(input_ids, labels)
                (loss / self.config.gradient_accumulation).backward()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Gradient step
            if (i + 1) % self.config.gradient_accumulation == 0:
                if self.scaler:
                    self.scaler.unscale_(optimizer)
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )
                grad_norms.append(grad_norm.item())
                
                if self.scaler:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
                self.global_step += 1
                
                # Checkpoint periódico
                if self.global_step % self.config.save_every_steps == 0:
                    self._save_checkpoint(
                        -1, optimizer, scheduler, total_loss / num_batches,
                        filename=f"step_{self.global_step}.pt"
                    )
            
            pbar.set_postfix({
                'loss': f'{total_loss/num_batches:.4f}',
                'grad': f'{grad_norms[-1]:.2f}' if grad_norms else 'N/A'
            })
        
        avg_loss = total_loss / num_batches
        metrics = {
            "grad_norm_mean": sum(grad_norms) / len(grad_norms) if grad_norms else 0,
            "grad_norm_max": max(grad_norms) if grad_norms else 0,
        }
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Evalúa el modelo."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            if self.scaler:
                with autocast('cuda'):
                    logits, loss, _ = self.model(input_ids, labels)
            else:
                logits, loss, _ = self.model(input_ids, labels)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, {"perplexity": perplexity}
    
    def _save_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        val_loss: float,
        filename: Optional[str] = None,
    ):
        """Guarda checkpoint."""
        if filename is None:
            filename = f"epoch_{epoch}.pt"
        
        path = self.output_dir / filename
        
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else vars(self.config),
        }, path)
        
        print(f"💾 Checkpoint: {path}")
        
        # Limpiar checkpoints antiguos
        self._cleanup_checkpoints()
    
    def _load_checkpoint(
        self,
        path: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ) -> int:
        """Carga checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        return checkpoint.get('epoch', 0)
    
    def _cleanup_checkpoints(self):
        """Mantiene solo los últimos N checkpoints."""
        checkpoints = sorted(
            self.output_dir.glob("epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for ckpt in checkpoints[self.config.keep_checkpoints:]:
            ckpt.unlink()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train PAMPAr-Coder 3B")
    
    parser.add_argument("--config", type=str, default="3B",
                        choices=["3B", "1.5B"])
    parser.add_argument("--data-dir", type=str, default="data/distillation")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/code_tokenizer.model")
    parser.add_argument("--output", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"🧠 PAMPAr-Coder {args.config}: Cloud Training")
    print(f"{'='*60}")
    
    # Config
    config = CONFIGS[args.config]
    
    # Model
    print("\n📦 Creando modelo...")
    model_config = ConfigPampaRCoderV2(
        vocab_size=config.vocab_size,
        dim=config.dim,
        n_heads=config.n_heads,
        n_capas=config.n_capas,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        use_mixed_precision=config.use_mixed_precision,
    )
    model = PampaRCoderV2(model_config)
    params = model.count_parameters()
    print(f"   Parámetros: {params['total']/1e9:.2f}B")
    
    # Dataset
    print("\n📚 Cargando datos...")
    data_files = list(Path(args.data_dir).glob("*.jsonl"))
    
    dataset = StreamingCodeDataset(
        data_paths=[str(f) for f in data_files],
        tokenizer_path=args.tokenizer,
        max_length=config.max_seq_len,
        max_samples=args.max_samples,
    )
    
    # Split
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=2,
    )
    
    print(f"   Train: {len(train_dataset)}")
    print(f"   Val: {len(val_dataset)}")
    
    # Trainer
    trainer = CloudTrainer(
        model=model,
        config=config,
        tokenizer_path=args.tokenizer,
        output_dir=args.output,
        use_wandb=not args.no_wandb,
    )
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
