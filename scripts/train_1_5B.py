# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr-Coder 1.5B: Entrenamiento cloud profesional.

Script optimizado para A40/A6000 (48GB VRAM):
- BF16 mixed precision + gradient checkpointing
- Cosine LR con warmup lineal
- Checkpoints por steps (no por epoch)
- Logging a archivo + resumable
- Todo el pipeline de datos (code + distillation)

Uso en RunPod:
    python train_1_5B.py                          # Entrenar desde cero
    python train_1_5B.py --resume checkpoint.pt   # Resumir
    python train_1_5B.py --eval-only              # Solo evaluar
"""

import os
import sys
import json
import math
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import sentencepiece as spm

# Agregar raíz del proyecto al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pampar.coder.v2 import crear_modelo, PRESET_1_5B, ConfigV2


# =============================================================================
# CONFIGURACIÓN DE ENTRENAMIENTO
# =============================================================================

TRAIN_CONFIG = {
    # Modelo
    "preset": "1_5B",
    "tokenizer": "data/tokenizer/pampar_48k.model",
    
    # Datos
    "data_dirs": [
        "data/code",
        "data/distillation",
    ],
    "max_seq_len": 4096,
    "val_split": 0.02,         # 2% para validación
    
    # Optimización
    "batch_size": 4,           # Per-GPU batch size (A40 48GB)
    "grad_accum": 8,           # Effective batch = 32
    "lr": 3e-4,
    "min_lr": 3e-5,            # 10% del LR máximo
    "warmup_steps": 2000,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "max_grad_norm": 1.0,
    "epochs": 3,
    
    # Checkpoints
    "save_every": 500,         # Guardar cada N steps
    "eval_every": 250,         # Evaluar cada N steps
    "log_every": 10,           # Log cada N steps
    "checkpoint_dir": "checkpoints_1_5B",
    "max_checkpoints": 5,      # Mantener solo los últimos N
    
    # Precisión
    "dtype": "bfloat16",       # bfloat16 para A40
}


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_dir: str):
    """Configura logging a archivo y consola."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/training_1_5B_{timestamp}.log"
    
    # Formato
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    
    return log_file


# =============================================================================
# DATASET — Streaming eficiente para corpus grande
# =============================================================================

class CodeDatasetStreaming(Dataset):
    """
    Dataset que carga JSONL en memoria con tokenización lazy.
    
    Para el 1.5B, necesitamos maximizar datos:
    - code/github_code.jsonl
    - code/train.jsonl
    - code/train_massive.jsonl
    - distillation/*.jsonl
    """
    
    def __init__(
        self,
        data_paths: list,
        tokenizer: spm.SentencePieceProcessor, 
        max_len: int = 4096,
        max_samples: int = 0,  # 0 = sin límite
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = []
        
        for path in data_paths:
            p = Path(path)
            if p.is_dir():
                for f in sorted(p.glob("*.jsonl")):
                    self._load_file(f)
            elif p.is_file():
                self._load_file(p)
        
        if max_samples > 0 and len(self.texts) > max_samples:
            import random
            random.shuffle(self.texts)
            self.texts = self.texts[:max_samples]
        
        logging.info(f"Dataset cargado: {len(self.texts):,} samples de {len(data_paths)} fuentes")
    
    def _load_file(self, path: Path):
        """Carga un archivo JSONL."""
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    text = self._extract_text(data)
                    if text and len(text) > 30:
                        self.texts.append(text)
                        count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        
        logging.info(f"  Cargado {path.name}: {count:,} samples")
    
    def _extract_text(self, data: dict) -> str:
        """Extrae texto de diferentes formatos JSONL."""
        # Instruction-following format
        if "instruction" in data and "output" in data:
            inp = data.get("input", "")
            inst = data["instruction"]
            out = data["output"]
            if inp:
                return f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            return f"### Instruction:\n{inst}\n\n### Response:\n{out}"
        
        # Code + docstring format
        if "code" in data:
            doc = data.get("docstring", "")
            code = data["code"]
            if doc:
                return f'"""{doc}"""\n{code}'
            return code
        
        # CommitPack / diff format
        if "new_contents" in data:
            msg = data.get("subject", data.get("message", ""))
            if msg:
                return f"# {msg}\n{data['new_contents']}"
            return data["new_contents"]
        
        # Generic text
        if "text" in data:
            return data["text"]
        
        return data.get("content", "")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenizar (con BOS/EOS)
        tokens = self.tokenizer.Encode(text)
        
        # Agregar BOS/EOS
        bos = self.tokenizer.bos_id()
        eos = self.tokenizer.eos_id()
        if bos >= 0:
            tokens = [bos] + tokens
        if eos >= 0:
            tokens = tokens + [eos]
        
        # Truncar a max_len + 1 (necesitamos 1 extra para labels)
        tokens = tokens[:self.max_len + 1]
        
        # Pad si es necesario
        pad_len = (self.max_len + 1) - len(tokens)
        if pad_len > 0:
            tokens = tokens + [0] * pad_len
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Input = tokens[:-1], Labels = tokens[1:]
        input_ids = tokens[:-1]
        labels = tokens[1:].clone()
        
        # Ignorar padding en loss
        labels[labels == 0] = -100
        
        return {"input_ids": input_ids, "labels": labels}


# =============================================================================
# LR SCHEDULER — Cosine con warmup
# =============================================================================

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate con warmup lineal."""
    # Warmup lineal
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    # Después del warmup: cosine decay
    if step >= max_steps:
        return min_lr
    
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# =============================================================================
# TRAINER PROFESIONAL
# =============================================================================

class Trainer1_5B:
    """Trainer optimizado para modelo 1.5B en A40."""
    
    def __init__(self, model, tokenizer, config: dict, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Mover a GPU
        self.model = self.model.to(device)
        
        # Registrar tokenizer para LLAVES
        n_mapped = model.registrar_tokenizer(tokenizer)
        logging.info(f"LLAVES: {n_mapped} tokens mapeados")
        
        # Mixed precision
        self.dtype = torch.bfloat16 if config["dtype"] == "bfloat16" else torch.float16
        self.scaler = None
        if self.dtype == torch.float16:
            self.scaler = torch.amp.GradScaler("cuda")
        
        # Optimizer — AdamW con parámetros tipo LLM
        param_groups = self._get_param_groups()
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config["lr"],
            betas=(config["beta1"], config["beta2"]),
            weight_decay=config["weight_decay"],
            fused=True,  # Faster on A40
        )
        
        # Estado
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []
        self.start_time = time.time()
    
    def _get_param_groups(self):
        """Separa params con/sin weight decay (no decay en biases y norms)."""
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        logging.info(f"Params con decay: {sum(p.numel() for p in decay_params):,}")
        logging.info(f"Params sin decay: {sum(p.numel() for p in no_decay_params):,}")
        
        return [
            {"params": decay_params, "weight_decay": self.config["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
    
    def train(self, train_loader, val_loader, max_steps: int):
        """Loop de entrenamiento principal."""
        cfg = self.config
        
        logging.info("=" * 70)
        logging.info("🧠 PAMPAr-Coder 1.5B — Entrenamiento")
        logging.info("=" * 70)
        logging.info(f"  Steps totales:      {max_steps:,}")
        logging.info(f"  Batch efectivo:     {cfg['batch_size'] * cfg['grad_accum']}")
        logging.info(f"  LR:                 {cfg['lr']} → {cfg['min_lr']} (cosine)")
        logging.info(f"  Warmup:             {cfg['warmup_steps']} steps")
        logging.info(f"  Dtype:              {cfg['dtype']}")
        logging.info(f"  Gradient accum:     {cfg['grad_accum']}")
        logging.info(f"  Save every:         {cfg['save_every']} steps")
        logging.info(f"  Eval every:         {cfg['eval_every']} steps")
        logging.info("=" * 70)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        accum_loss = 0.0
        accum_count = 0
        epoch = 0
        
        train_iter = iter(train_loader)
        
        while self.global_step < max_steps:
            # Obtener batch (cycling through epochs)
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                logging.info(f"--- Epoch {epoch} completada ---")
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            # Forward + backward
            loss = self._train_step(batch)
            accum_loss += loss
            accum_count += 1
            
            # Optimizer step (cada grad_accum micro-batches)
            if accum_count >= cfg["grad_accum"]:
                self._optimizer_step()
                
                avg_loss = accum_loss / accum_count
                self.train_losses.append(avg_loss)
                
                # Logging
                if self.global_step % cfg["log_every"] == 0:
                    self._log_step(avg_loss)
                
                # Evaluación
                if self.global_step % cfg["eval_every"] == 0 and self.global_step > 0:
                    val_loss = self._evaluate(val_loader)
                    self.val_losses.append((self.global_step, val_loss))
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint("best")
                        logging.info(f"  ✨ Nuevo mejor modelo! val_loss={val_loss:.4f}")
                    
                    self.model.train()
                
                # Checkpoint
                if self.global_step % cfg["save_every"] == 0 and self.global_step > 0:
                    self._save_checkpoint(f"step_{self.global_step}")
                    self._cleanup_checkpoints()
                
                accum_loss = 0.0
                accum_count = 0
        
        # Final
        self._save_checkpoint("final")
        logging.info(f"✅ Entrenamiento completo! Best val_loss: {self.best_val_loss:.4f}")
    
    def _train_step(self, batch) -> float:
        """Un micro-batch forward+backward."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        with torch.amp.autocast("cuda", dtype=self.dtype):
            _, loss, _ = self.model(input_ids, targets=labels)
            loss = loss / self.config["grad_accum"]
        
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config["grad_accum"]
    
    def _optimizer_step(self):
        """Clip gradients + optimizer step + LR update."""
        cfg = self.config
        
        # Unscale si usamos FP16
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
        
        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), cfg["max_grad_norm"]
        )
        
        # LR scheduling
        max_steps = self._estimate_max_steps()
        lr = get_lr(
            self.global_step, cfg["warmup_steps"], max_steps,
            cfg["lr"], cfg["min_lr"]
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        
        # Step
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        
        return grad_norm.item()
    
    def _estimate_max_steps(self) -> int:
        """Estima steps totales basado en config."""
        # Será sobreescrito cuando conocemos el dataset size
        return getattr(self, "_max_steps", 100000)
    
    @torch.no_grad()
    def _evaluate(self, val_loader, max_batches: int = 50) -> float:
        """Evalúa en validation set."""
        self.model.eval()
        total_loss = 0
        count = 0
        
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            with torch.amp.autocast("cuda", dtype=self.dtype):
                _, loss, _ = self.model(input_ids, targets=labels)
            
            total_loss += loss.item()
            count += 1
        
        avg_loss = total_loss / max(count, 1)
        ppl = math.exp(min(avg_loss, 20))
        
        logging.info(
            f"  📊 EVAL step={self.global_step:,} | "
            f"val_loss={avg_loss:.4f} | ppl={ppl:.1f}"
        )
        
        return avg_loss
    
    def _log_step(self, loss: float):
        """Log de progreso."""
        elapsed = time.time() - self.start_time
        steps_per_sec = max(self.global_step, 1) / elapsed
        
        max_steps = getattr(self, "_max_steps", 0)
        remaining = (max_steps - self.global_step) / max(steps_per_sec, 1e-6)
        eta = timedelta(seconds=int(remaining))
        
        lr = self.optimizer.param_groups[0]["lr"]
        
        # GPU memory
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1024**3
            mem_str = f" | GPU={mem:.1f}GB"
        else:
            mem_str = ""
        
        logging.info(
            f"  step={self.global_step:>7,}/{max_steps:,} | "
            f"loss={loss:.4f} | lr={lr:.2e} | "
            f"{steps_per_sec:.2f} steps/s | ETA={eta}{mem_str}"
        )
    
    def _save_checkpoint(self, name: str):
        """Guarda checkpoint completo."""
        ckpt_dir = Path(self.config["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        path = ckpt_dir / f"{name}.pt"
        
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses[-1000:],  # Últimos 1000
            "val_losses": self.val_losses,
            "config": {
                "model": "PRESET_1_5B",
                "training": self.config,
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        if self.scaler:
            ckpt["scaler"] = self.scaler.state_dict()
        
        torch.save(ckpt, path)
        logging.info(f"💾 Checkpoint guardado: {path} ({path.stat().st_size / 1024**3:.1f}GB)")
    
    def _cleanup_checkpoints(self):
        """Mantiene solo los últimos N checkpoints (no borra best/final)."""
        ckpt_dir = Path(self.config["checkpoint_dir"])
        
        # Listar checkpoints por step
        step_ckpts = sorted(
            [f for f in ckpt_dir.glob("step_*.pt")],
            key=lambda f: f.stat().st_mtime,
        )
        
        # Borrar los más viejos
        max_keep = self.config["max_checkpoints"]
        while len(step_ckpts) > max_keep:
            old = step_ckpts.pop(0)
            old.unlink()
            logging.info(f"🗑️  Checkpoint eliminado: {old.name}")
    
    def load_checkpoint(self, path: str):
        """Carga checkpoint para resumir."""
        logging.info(f"🔄 Cargando checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.train_losses = ckpt.get("train_losses", [])
        self.val_losses = ckpt.get("val_losses", [])
        
        if self.scaler and "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        
        logging.info(f"  Resumido desde step {self.global_step:,}, best_val={self.best_val_loss:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PAMPAr-Coder 1.5B Training")
    parser.add_argument("--resume", type=str, help="Checkpoint para resumir")
    parser.add_argument("--eval-only", action="store_true", help="Solo evaluar")
    parser.add_argument("--max-steps", type=int, default=0, help="Override max steps (0=auto)")
    parser.add_argument("--batch-size", type=int, default=0, help="Override batch size")
    parser.add_argument("--lr", type=float, default=0, help="Override learning rate")
    parser.add_argument("--data", nargs="+", help="Override data directories")
    args = parser.parse_args()
    
    # Config
    cfg = TRAIN_CONFIG.copy()
    if args.batch_size > 0:
        cfg["batch_size"] = args.batch_size
    if args.lr > 0:
        cfg["lr"] = args.lr
    if args.data:
        cfg["data_dirs"] = args.data
    
    # Logging
    log_file = setup_logging(cfg["checkpoint_dir"])
    logging.info(f"Log: {log_file}")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        logging.info(f"GPU: {gpu_name} ({gpu_mem:.0f}GB)")
    else:
        logging.warning("⚠️  CUDA no disponible, entrenando en CPU (MUY lento)")
    
    # Tokenizer
    tokenizer_path = str(PROJECT_ROOT / cfg["tokenizer"])
    logging.info(f"Tokenizer: {tokenizer_path}")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(tokenizer_path)
    logging.info(f"  Vocab: {tokenizer.GetPieceSize():,} tokens")
    
    # Verificar que el vocab coincide con el preset
    assert tokenizer.GetPieceSize() == PRESET_1_5B.vocab_size, (
        f"Vocab mismatch! Tokenizer={tokenizer.GetPieceSize()}, "
        f"Config={PRESET_1_5B.vocab_size}"
    )
    
    # Modelo
    logging.info("Creando modelo PRESET_1_5B...")
    model = crear_modelo(PRESET_1_5B)
    
    params = model.count_params()
    logging.info(f"  Parámetros: {params['total']:,} ({params['total']/1e9:.2f}B)")
    logging.info(f"  Memoria FP16: {params['total'] * 2 / 1024**2:.0f}MB")
    logging.info(f"  Memoria BF16: {params['total'] * 2 / 1024**2:.0f}MB")
    
    # Dataset
    data_paths = [str(PROJECT_ROOT / d) for d in cfg["data_dirs"]]
    logging.info(f"Cargando datos de: {cfg['data_dirs']}")
    
    full_dataset = CodeDatasetStreaming(
        data_paths, tokenizer, max_len=PRESET_1_5B.max_seq_len
    )
    
    # Split train/val
    n_val = max(int(len(full_dataset) * cfg["val_split"]), 100)
    n_train = len(full_dataset) - n_val
    
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=generator
    )
    
    logging.info(f"  Train: {n_train:,} | Val: {n_val:,}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    # Calcular steps totales
    steps_per_epoch = len(train_loader) // cfg["grad_accum"]
    max_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * cfg["epochs"]
    
    logging.info(f"  Steps/epoch: {steps_per_epoch:,}")
    logging.info(f"  Total steps: {max_steps:,}")
    logging.info(f"  Epochs: {cfg['epochs']}")
    
    # Trainer
    trainer = Trainer1_5B(model, tokenizer, cfg, device=device)
    trainer._max_steps = max_steps
    
    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Eval only
    if args.eval_only:
        val_loss = trainer._evaluate(val_loader, max_batches=200)
        logging.info(f"Val loss: {val_loss:.4f} | PPL: {math.exp(val_loss):.1f}")
        return
    
    # Train!
    trainer.train(train_loader, val_loader, max_steps)
    
    # Resumen final
    logging.info("\n" + "=" * 70)
    logging.info("📊 RESUMEN FINAL")
    logging.info("=" * 70)
    logging.info(f"  Steps completados: {trainer.global_step:,}")
    logging.info(f"  Mejor val_loss:    {trainer.best_val_loss:.4f}")
    if trainer.train_losses:
        logging.info(f"  Último train_loss: {trainer.train_losses[-1]:.4f}")
    elapsed = time.time() - trainer.start_time
    logging.info(f"  Tiempo total:      {timedelta(seconds=int(elapsed))}")
    logging.info(f"  Checkpoints en:    {cfg['checkpoint_dir']}/")


if __name__ == "__main__":
    main()
