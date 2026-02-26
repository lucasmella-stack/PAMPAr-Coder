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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
import sentencepiece as spm
from tqdm import tqdm

# 8-bit AdamW para reducir memoria del optimizer ~4x
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

# Agregar path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pampar.coder.v2.modelo import PampaRCoderV2
from pampar.coder.v2.config import ConfigV2, PRESET_1_5B
from pampar.coder.v2.zonas import Territorio
from pampar.coder.v2.aprendizaje.memoria_jerarquica import MemoriaJerarquica
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
        model_config: ConfigV2 = None,
        use_memoria: bool = True,
        memoria_config: Optional[Dict] = None,
    ):
        self.model = model
        self.config = config
        self.model_config = model_config
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
        self.scaler = GradScaler('cuda') if config.use_amp else None
        
        # --- Memoria Jerárquica Pareto ---
        self.use_memoria = use_memoria
        if use_memoria:
            mem_cfg = memoria_config or {}
            self.memoria = MemoriaJerarquica(
                capacidad_l0=mem_cfg.get("capacidad_l0", 4096),
                capacidad_l1=mem_cfg.get("capacidad_l1", 10000),
                capacidad_l2=mem_cfg.get("capacidad_l2", 5000),
                ventana_tokens=mem_cfg.get("ventana", 16),
                umbral_loss_alta=mem_cfg.get("umbral_loss", 3.0),
                lr_interiorizacion=mem_cfg.get("lr_interiorizacion", 1e-5),
            )
            self.replay_every = mem_cfg.get("replay_every", 100)
            self.consolidar_every = mem_cfg.get("consolidar_every", 500)
            self.replay_batch_size = mem_cfg.get("replay_batch_size", 8)
            print("   🧠 Memoria Jerárquica activada ")
        else:
            self.memoria = None
        
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
        
        # Optimizer — 8-bit AdamW si disponible (reduce memoria ~4x)
        if HAS_BNB:
            print("   ⚡ Usando AdamW8bit (bitsandbytes)")
            optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.95),
            )
        else:
            print("   ⚠️  bitsandbytes no disponible, usando AdamW estándar")
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
                    logits, loss, info = self.model(input_ids, labels)
                self.scaler.scale(loss / self.config.gradient_accumulation).backward()
            else:
                logits, loss, info = self.model(input_ids, labels)
                (loss / self.config.gradient_accumulation).backward()
            
            total_loss += loss.item()
            num_batches += 1
            
            # --- Memoria Jerárquica: capturar patrones difíciles ---
            if self.use_memoria and self.memoria is not None:
                with torch.no_grad():
                    # Per-token loss para identificar dónde falla el modelo
                    per_token_loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        labels[:, 1:].reshape(-1),
                        ignore_index=-100,
                        reduction='none',
                    ).reshape(input_ids.size(0), -1)  # [B, L-1]
                    
                    # Pad para que coincida con input_ids shape [B, L]
                    pad = torch.zeros(
                        input_ids.size(0), 1,
                        device=per_token_loss.device,
                    )
                    per_token_loss = torch.cat([pad, per_token_loss], dim=1)
                    
                    terr_acts = info.get("terr_acts")
                    self.memoria.procesar_batch(
                        input_ids=input_ids,
                        per_token_loss=per_token_loss,
                        terr_acts=terr_acts,
                    )
            
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
                
                # --- Memoria: replay de patrones difíciles ---
                if (
                    self.use_memoria
                    and self.memoria is not None
                    and self.global_step % self.replay_every == 0
                ):
                    self._replay_step(optimizer)
                
                # --- Memoria: consolidación periódica ---
                if (
                    self.use_memoria
                    and self.memoria is not None
                    and self.global_step % self.consolidar_every == 0
                ):
                    consolidacion = self.memoria.consolidar(model=self.model)
                    if self.use_wandb:
                        mem_stats = self.memoria.stats()
                        self.wandb.log({
                            "memoria/l0_uso": mem_stats["niveles"]["l0"]["uso_pct"],
                            "memoria/l1_uso": mem_stats["niveles"]["l1"]["uso_pct"],
                            "memoria/l2_uso": mem_stats["niveles"]["l2"]["uso_pct"],
                            "memoria/interiorizados_l3": mem_stats["total_interiorizados_l3"],
                            "memoria/tokens_procesados": mem_stats["total_tokens_procesados"],
                            "memoria/compresion_pct": mem_stats["ratio_compresion_efectiva"],
                        }, step=self.global_step)
                    print(f"   🧠 Consolidación @ step {self.global_step}: {self.memoria}")
                
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
        
        checkpoint_data = {
            'model': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'global_step': self.global_step,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'train_config': asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else vars(self.config),
            'model_config': asdict(self.model_config) if self.model_config else None,
        }
        torch.save(checkpoint_data, path)
        
        # Guardar memoria jerárquica por separado (JSON, más portable)
        if self.use_memoria and self.memoria is not None:
            mem_path = path.with_suffix('.memoria.json')
            self.memoria.guardar(str(mem_path))
        
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
        
        # Restaurar memoria jerárquica si existe
        if self.use_memoria:
            mem_path = Path(path).with_suffix('.memoria.json')
            if mem_path.exists():
                self.memoria = MemoriaJerarquica.cargar(str(mem_path))
                print(f"   🧠 Memoria restaurada: {self.memoria}")
        
        return checkpoint.get('epoch', 0)
    
    def _replay_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Entrena un micro-step con patrones difíciles de la memoria."""
        assert self.memoria is not None

        replay_batch = self.memoria.get_replay_batch(
            batch_size=self.replay_batch_size,
            nivel="l1",
            strategy="hardest",
        )
        if replay_batch is None:
            return

        replay_ids = replay_batch.to(self.device)
        # Targets = shifted input_ids
        replay_targets = replay_ids.clone()
        replay_targets[:, :-1] = replay_ids[:, 1:]
        replay_targets[:, -1] = -100

        self.model.train()
        if self.scaler:
            with autocast('cuda'):
                _, replay_loss, _ = self.model(replay_ids, replay_targets)
            if replay_loss is not None:
                self.scaler.scale(replay_loss * 0.1).backward()
        else:
            _, replay_loss, _ = self.model(replay_ids, replay_targets)
            if replay_loss is not None:
                (replay_loss * 0.1).backward()

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
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/pampar_48k.model")
    parser.add_argument("--output", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-memoria", action="store_true",
                        help="Desactivar memoria jerárquica Pareto")
    parser.add_argument("--mem-replay-every", type=int, default=100,
                        help="Steps entre replay batches")
    parser.add_argument("--mem-consolidar-every", type=int, default=500,
                        help="Steps entre consolidaciones")
    parser.add_argument("--mem-umbral-loss", type=float, default=3.0,
                        help="Loss mínima para considerar patrón importante")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"🧠 PAMPAr-Coder {args.config}: Cloud Training")
    print(f"{'='*60}")
    
    # Config
    config = CONFIGS[args.config]
    
    # Model config — PRESET_1_5B para modelo real, Config3B para 3B
    if args.config == "1.5B":
        model_config = PRESET_1_5B
    else:
        model_config = ConfigV2(
            vocab_size=config.vocab_size,
            dim=config.dim,
            n_heads=config.n_heads,
            n_capas=config.n_capas,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            use_checkpoint=config.use_gradient_checkpointing,
            use_amp=config.use_mixed_precision,
        )
    
    # Model
    print("\n📦 Creando modelo...")
    model = PampaRCoderV2(model_config)
    params = model.count_params()
    print(f"   Parámetros: {params['total']/1e9:.2f}B")
    print(f"   Arquitectura: dim={model_config.dim}, capas={model_config.n_capas}, "
          f"heads={model_config.n_heads}Q/{model_config.kv_heads}KV")
    print(f"   Vocab: {model_config.vocab_size}, Seq: {config.max_seq_len}")
    print(f"   Nuevas features v2.1: ventana_contexto={model_config.ventana_contexto}, "
          f"sym_factor={model_config.sym_factor}, exit_percentile={model_config.exit_percentile}")
    
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
    
    # Config de memoria jerárquica
    memoria_config = {
        "replay_every": args.mem_replay_every,
        "consolidar_every": args.mem_consolidar_every,
        "umbral_loss": args.mem_umbral_loss,
    }
    
    # Trainer
    trainer = CloudTrainer(
        model=model,
        config=config,
        tokenizer_path=args.tokenizer,
        output_dir=args.output,
        use_wandb=not args.no_wandb,
        model_config=model_config,
        use_memoria=not args.no_memoria,
        memoria_config=memoria_config,
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
