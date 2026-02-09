# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr-Coder v2: Entrenamiento optimizado.

Componentes:
- CodeDataset: Dataset eficiente con streaming
- Trainer: Loop de entrenamiento simple
- Checkpointing: Guardar/resumir entrenamiento

Uso:
    python scripts/train_v2_refactored.py --preset 4GB --epochs 10
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.coder.v2 import crear_modelo, PRESET_4GB, PRESET_8GB


# =============================================================================
# DATASET
# =============================================================================

class CodeDataset(Dataset):
    """Dataset de código ligero y eficiente."""
    
    def __init__(self, data_dir: str, tokenizer: spm.SentencePieceProcessor, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        
        # Cargar todos los JSONL
        data_path = Path(data_dir)
        for f in sorted(data_path.glob("*.jsonl")):
            self._load_file(f)
        
        print(f"📚 Dataset: {len(self.samples):,} samples")
    
    def _load_file(self, path: Path):
        """Carga un archivo JSONL."""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = self._extract_text(data)
                    if text and len(text) > 20:
                        self.samples.append(text)
                except:
                    continue
    
    def _extract_text(self, data: dict) -> str:
        """Extrae texto de diferentes formatos."""
        # Instruction/output format
        if "instruction" in data and "output" in data:
            return f"[INST]{data['instruction']}[/INST]{data['output']}"
        
        # Code/docstring format
        if "code" in data:
            doc = data.get("docstring", "")
            return f"{doc}\n{data['code']}" if doc else data["code"]
        
        # CommitPack format
        if "new_contents" in data:
            return data["new_contents"]
        
        return data.get("content", "")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        
        # Tokenizar
        tokens = self.tokenizer.Encode(text)[:self.max_len]
        
        # Pad
        pad_len = self.max_len - len(tokens)
        if pad_len > 0:
            tokens = tokens + [0] * pad_len
        
        ids = torch.tensor(tokens, dtype=torch.long)
        
        # Labels = shifted input
        labels = ids.clone()
        labels[:-1] = ids[1:]
        labels[-1] = -100
        
        return {"input_ids": ids, "labels": labels}


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """Trainer simple y eficiente."""
    
    def __init__(self, model, tokenizer, device="cuda", use_amp=True):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.use_amp = use_amp and device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        
        # Registrar tokenizer
        model.registrar_tokenizer(tokenizer)
        
        # Historial
        self.history = {"train_loss": [], "val_loss": [], "epoch": []}
    
    def train_epoch(self, loader, optimizer, grad_accum=4):
        """Entrena una epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        optimizer.zero_grad()
        
        for i, batch in enumerate(loader):
            ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    _, loss, _ = self.model(ids, labels)
                self.scaler.scale(loss / grad_accum).backward()
            else:
                _, loss, _ = self.model(ids, labels)
                (loss / grad_accum).backward()
            
            total_loss += loss.item()
            n_batches += 1
            
            # Step
            if (i + 1) % grad_accum == 0:
                if self.use_amp:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def evaluate(self, loader):
        """Evalúa el modelo."""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        for batch in loader:
            ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    _, loss, _ = self.model(ids, labels)
            else:
                _, loss, _ = self.model(ids, labels)
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def save(self, path: str, epoch: int, optimizer=None, val_loss=None):
        """Guarda checkpoint."""
        ckpt = {
            "model": self.model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "history": self.history,
        }
        if optimizer:
            ckpt["optimizer"] = optimizer.state_dict()
        
        torch.save(ckpt, path)
        print(f"💾 Saved: {path}")
    
    def load(self, path: str, optimizer=None):
        """Carga checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        
        if optimizer and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        
        if "history" in ckpt:
            self.history = ckpt["history"]
        
        return ckpt.get("epoch", 0)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/code")
    parser.add_argument("--tokenizer", default="data/tokenizer/code.model")
    parser.add_argument("--preset", default="4GB", choices=["4GB", "8GB"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧠 PAMPAr-Coder v2 Training")
    print("=" * 60)
    
    # Directorio output
    Path(args.output).mkdir(exist_ok=True)
    
    # Tokenizer
    print(f"\n📖 Loading tokenizer: {args.tokenizer}")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(args.tokenizer)
    
    # Modelo
    print(f"\n📦 Creating model (preset: {args.preset})")
    config = PRESET_4GB if args.preset == "4GB" else PRESET_8GB
    model = crear_modelo(config)
    
    params = model.count_params()
    print(f"   Params: {params['total']:,}")
    print(f"   Memory: {params['total'] * 2 / 1024**2:.1f} MB (FP16)")
    
    # Dataset
    print(f"\n📚 Loading data: {args.data}")
    dataset = CodeDataset(args.data, tokenizer, max_len=config.max_seq_len)
    
    # Split
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    
    print(f"   Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    
    # Trainer
    trainer = Trainer(model, tokenizer, device=args.device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Resume
    start_epoch = 0
    if args.resume:
        print(f"\n🔄 Resuming from {args.resume}")
        start_epoch = trainer.load(args.resume, optimizer)
    
    # Training
    print(f"\n🚀 Training for {args.epochs} epochs...")
    print("-" * 60)
    
    best_loss = float("inf")
    
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer, args.grad_accum)
        
        # Eval
        val_loss = trainer.evaluate(val_loader)
        
        # Log
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"PPL: {torch.exp(torch.tensor(val_loss)):.1f} | "
              f"Time: {elapsed:.0f}s")
        
        # History
        trainer.history["train_loss"].append(train_loss)
        trainer.history["val_loss"].append(val_loss)
        trainer.history["epoch"].append(epoch + 1)
        
        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            trainer.save(f"{args.output}/pampar_v2_best.pt", epoch + 1, optimizer, val_loss)
            print(f"   ✨ New best!")
        
        # Save periodic
        if (epoch + 1) % 5 == 0:
            trainer.save(f"{args.output}/pampar_v2_epoch_{epoch+1}.pt", epoch + 1, optimizer, val_loss)
    
    # Final
    trainer.save(f"{args.output}/pampar_v2_final.pt", args.epochs, optimizer, val_loss)
    
    print("\n" + "=" * 60)
    print(f"✅ Training complete! Best val loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
