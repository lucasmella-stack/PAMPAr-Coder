# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr-Coder v2: Script de Entrenamiento.

Entrena el modelo con 52 zonas de Brodmann usando:
1. Pre-training en código existente
2. Knowledge Distillation (opcional)
3. Fine-tuning con datos de alta calidad

Optimizado para GTX 1650 (4GB VRAM).
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import sentencepiece as spm

# Agregar path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.coder.model_v2 import PampaRCoderV2, ConfigPampaRCoderV2, CODER_V2_4GB
from pampar.coder.llaves_codigo import clasificar_token, TipoTokenCodigo


# =============================================================================
# DATASET
# =============================================================================

class CodeDataset(Dataset):
    """Dataset de código para entrenamiento."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        max_length: int = 512,
        max_samples: Optional[int] = None,
    ):
        self.max_length = max_length
        
        # Cargar tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        
        # Cargar datos
        self.samples = []
        data_path = Path(data_path)
        
        if data_path.is_file():
            self._load_file(data_path, max_samples)
        elif data_path.is_dir():
            for f in sorted(data_path.glob("*.jsonl")):
                self._load_file(f, max_samples)
                if max_samples and len(self.samples) >= max_samples:
                    break
        
        print(f"📚 Loaded {len(self.samples)} samples")
    
    def _load_file(self, path: Path, max_samples: Optional[int]):
        """Carga un archivo JSONL."""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_samples and len(self.samples) >= max_samples:
                    break
                
                try:
                    data = json.loads(line)
                    
                    # Extraer texto
                    text = self._extract_text(data)
                    if text and len(text) > 50:
                        self.samples.append(text)
                except:
                    continue
    
    def _extract_text(self, data: Dict) -> str:
        """Extrae texto de diferentes formatos."""
        # Formato instruction-response
        if "instruction" in data and "response" in data:
            instruction = data.get("instruction", "")
            response = data.get("response", "") or data.get("output", "")
            if response:
                return f"[INST]{instruction}[/INST]{response}"
        
        # Formato input-output
        if "input" in data and "output" in data:
            return f"[INST]{data['input']}[/INST]{data['output']}"
        
        # Formato code
        if "code" in data:
            return data["code"]
        
        # Formato content
        if "content" in data:
            return data["content"]
        
        return ""
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        
        # Tokenizar
        tokens = self.tokenizer.Encode(text)[:self.max_length]
        
        # Padding
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # Labels (shifted)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


# =============================================================================
# TRAINER
# =============================================================================

class TrainerV2:
    """Trainer para PAMPAr-Coder v2."""
    
    def __init__(
        self,
        model: PampaRCoderV2,
        tokenizer_path: str,
        device: str = "cuda",
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and device == "cuda"
        
        # Tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        
        # Registrar en modelo
        model.registrar_tokenizer(self.tokenizer)
        
        # Scaler para AMP
        self.scaler = GradScaler() if self.use_amp else None
        
        # Métricas
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "epoch": [],
        }
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        gradient_accumulation: int = 1,
        max_grad_norm: float = 1.0,
    ) -> float:
        """Entrena una epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        optimizer.zero_grad()
        
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward
            if self.use_amp:
                with autocast(device_type='cuda'):
                    logits, loss, _ = self.model(input_ids, labels)
                
                # Backward
                self.scaler.scale(loss / gradient_accumulation).backward()
            else:
                logits, loss, _ = self.model(input_ids, labels)
                (loss / gradient_accumulation).backward()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Gradient step
            if (i + 1) % gradient_accumulation == 0:
                if self.use_amp:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
                
                if scheduler:
                    scheduler.step()
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evalúa el modelo."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            if self.use_amp:
                with autocast(device_type='cuda'):
                    logits, loss, _ = self.model(input_ids, labels)
            else:
                logits, loss, _ = self.model(input_ids, labels)
            
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += (labels != -100).sum().item()
        
        avg_loss = total_loss / len(dataloader.dataset)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
        }
    
    @torch.no_grad()
    def generate_sample(self, prompt: str, max_tokens: int = 100) -> str:
        """Genera código de ejemplo."""
        self.model.eval()
        
        # Tokenizar
        tokens = self.tokenizer.Encode(prompt)
        input_ids = torch.tensor([tokens], device=self.device)
        
        # Generar
        generated, stats = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_k=50,
        )
        
        # Decodificar
        output = self.tokenizer.Decode(generated[0].tolist())
        
        return output
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer: torch.optim.Optimizer = None,
        val_loss: float = None,
    ):
        """Guarda checkpoint."""
        checkpoint = {
            "model": self.model.state_dict(),
            "config": vars(self.model.config),
            "epoch": epoch,
            "val_loss": val_loss,
            "history": self.history,
        }
        
        if optimizer:
            checkpoint["optimizer"] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved: {path}")
    
    def load_checkpoint(
        self,
        path: str,
        optimizer: torch.optim.Optimizer = None,
    ) -> int:
        """Carga checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model"])
        
        if optimizer and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        
        return checkpoint.get("epoch", 0)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train PAMPAr-Coder v2")
    
    # Data
    parser.add_argument("--data", type=str, default="data/distillation",
                        help="Path to training data")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/code.model",
                        help="Path to tokenizer")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max training samples")
    
    # Model
    parser.add_argument("--preset", type=str, default="4GB",
                        choices=["4GB", "8GB", "24GB"],
                        help="Model preset")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    
    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    
    # Output
    parser.add_argument("--output", type=str, default="checkpoints",
                        help="Output directory")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision")
    
    args = parser.parse_args()
    
    # ===========================================================================
    print("=" * 70)
    print("🧠 PAMPAr-Coder v2: Training Script")
    print("   52 Zonas de Brodmann | Knowledge Distillation Ready")
    print("=" * 70)
    
    # Crear directorio output
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # ===========================================================================
    # MODELO
    # ===========================================================================
    print(f"\n📦 Creating model (preset: {args.preset})...")
    
    if args.preset == "4GB":
        config = CODER_V2_4GB
    elif args.preset == "8GB":
        from pampar.coder.model_v2 import CODER_V2_8GB
        config = CODER_V2_8GB
    else:
        from pampar.coder.model_v2 import CODER_V2_24GB
        config = CODER_V2_24GB
    
    model = PampaRCoderV2(config)
    params = model.count_parameters()
    print(f"   Parameters: {params['total']:,}")
    print(f"   Memory: {params['total'] * 2 / 1024**2:.1f} MB (FP16)")
    
    # ===========================================================================
    # TRAINER
    # ===========================================================================
    print(f"\n🎯 Initializing trainer...")
    
    trainer = TrainerV2(
        model=model,
        tokenizer_path=args.tokenizer,
        device=args.device,
        use_amp=not args.no_amp,
    )
    
    # ===========================================================================
    # DATASET
    # ===========================================================================
    print(f"\n📚 Loading dataset from {args.data}...")
    
    dataset = CodeDataset(
        data_path=args.data,
        tokenizer_path=args.tokenizer,
        max_length=config.max_seq_len,
        max_samples=args.max_samples,
    )
    
    # Split
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    
    # ===========================================================================
    # OPTIMIZER
    # ===========================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Scheduler con warmup
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        progress = (step - args.warmup_steps) / (total_steps - args.warmup_steps)
        return 0.1 + 0.9 * (1 - progress)  # Cosine decay to 10%
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # ===========================================================================
    # RESUME
    # ===========================================================================
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\n🔄 Resuming from {args.resume}...")
        start_epoch = trainer.load_checkpoint(args.resume, optimizer)
        print(f"   Resumed at epoch {start_epoch}")
    
    # ===========================================================================
    # TRAINING LOOP
    # ===========================================================================
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Gradient accumulation: {args.grad_accum}")
    print(f"   Effective batch: {args.batch_size * args.grad_accum}")
    print("-" * 70)
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = trainer.train_epoch(
            train_loader,
            optimizer,
            scheduler,
            gradient_accumulation=args.grad_accum,
        )
        
        # Evaluate
        val_metrics = trainer.evaluate(val_loader)
        val_loss = val_metrics["loss"]
        
        # Time
        epoch_time = time.time() - epoch_start
        
        # Log
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | "
              f"PPL: {val_metrics['perplexity']:.2f} | "
              f"Time: {epoch_time:.1f}s")
        
        # History
        trainer.history["train_loss"].append(train_loss)
        trainer.history["val_loss"].append(val_loss)
        trainer.history["epoch"].append(epoch + 1)
        trainer.history["lr"].append(optimizer.param_groups[0]["lr"])
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            path = Path(args.output) / f"pampar_v2_epoch_{epoch+1}.pt"
            trainer.save_checkpoint(str(path), epoch + 1, optimizer, val_loss)
        
        # Best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = Path(args.output) / "pampar_v2_best.pt"
            trainer.save_checkpoint(str(path), epoch + 1, optimizer, val_loss)
            print(f"   ✨ New best model saved!")
        
        # Generate sample
        if (epoch + 1) % 2 == 0:
            sample = trainer.generate_sample(
                "[INST]Write a function to calculate factorial[/INST]",
                max_tokens=100
            )
            print(f"\n   📝 Sample:\n   {sample[:200]}...")
            print()
    
    print("=" * 70)
    print("✅ Training complete!")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoints saved to: {args.output}")


if __name__ == "__main__":
    main()
