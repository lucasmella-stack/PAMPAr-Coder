# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Entrenamiento en modo BACKGROUND para PAMPAr-Coder.

Características:
- Prioridad baja (no interfiere con tu trabajo)
- Checkpoints automáticos cada N steps
- Resume automático si se interrumpe
- Logging mínimo (no spamea la consola)
- Usa GPU solo cuando está idle
"""

import os
import sys
import json
import time
import argparse
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler

# Configurar prioridad baja en Windows
if sys.platform == 'win32':
    import ctypes
    # BELOW_NORMAL_PRIORITY_CLASS = 0x4000
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), 
        0x4000  # BELOW_NORMAL_PRIORITY_CLASS
    )

sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.coder import crear_modelo, CODER_4GB

# Silenciar warnings
import warnings
warnings.filterwarnings('ignore')


class CodeDataset(Dataset):
    """Dataset simple para código."""
    
    def __init__(self, data_path: Path, tokenizer, max_length: int = 512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Cargar datos
        if data_path.suffix == '.jsonl':
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        text = item.get('text', item.get('code', ''))
                        if text and len(text) > 20:
                            self.samples.append(text)
                    except:
                        continue
        
        print(f"Dataset cargado: {len(self.samples):,} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        
        # Tokenizar
        tokens = self.tokenizer.encode(text)
        
        # Pad/truncate
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens[:-1], tokens[1:]  # input, target


class BackgroundTrainer:
    """Entrenador que corre en background sin molestar."""
    
    def __init__(
        self,
        model: nn.Module,
        train_data: DataLoader,
        checkpoint_dir: Path,
        device: torch.device,
        lr: float = 1e-4,
        checkpoint_every: int = 500,
        log_every: int = 100,
    ):
        self.model = model.to(device)
        self.train_data = train_data
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.scaler = GradScaler('cuda') if device.type == 'cuda' else None
        
        self.global_step = 0
        self.total_loss = 0
        self.start_time = None
        self.running = True
        
        # Status file para monitoreo externo
        self.status_file = self.checkpoint_dir / "training_status.json"
        
        # Signal handler para interrupciones
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        print("\n⚠️  Interrupción detectada, guardando checkpoint...")
        self.running = False
    
    def _save_checkpoint(self, epoch: int, loss: float):
        """Guarda checkpoint para resume."""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': epoch,
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Guardar checkpoint
        ckpt_path = self.checkpoint_dir / "background_latest.pt"
        torch.save(checkpoint, ckpt_path)
        
        # También guardar el mejor
        best_path = self.checkpoint_dir / "background_best.pt"
        if not best_path.exists():
            torch.save(checkpoint, best_path)
        else:
            best = torch.load(best_path, weights_only=False)
            if loss < best.get('loss', float('inf')):
                torch.save(checkpoint, best_path)
    
    def _update_status(self, epoch: int, batch: int, loss: float, speed: float):
        """Actualiza archivo de status para monitoreo."""
        elapsed = time.time() - self.start_time
        status = {
            'running': self.running,
            'epoch': epoch,
            'batch': batch,
            'global_step': self.global_step,
            'loss': round(loss, 4),
            'speed': round(speed, 2),
            'elapsed_hours': round(elapsed / 3600, 2),
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    def load_checkpoint(self) -> bool:
        """Intenta cargar checkpoint existente."""
        ckpt_path = self.checkpoint_dir / "background_latest.pt"
        
        if ckpt_path.exists():
            print(f"📂 Resumiendo desde checkpoint...")
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.global_step = checkpoint['global_step']
            print(f"   Step: {self.global_step}, Loss: {checkpoint['loss']:.4f}")
            return True
        return False
    
    def train_epoch(self, epoch: int):
        """Entrena una época."""
        self.model.train()
        epoch_loss = 0
        n_batches = 0
        batch_times = []
        
        for batch_idx, (input_ids, targets) in enumerate(self.train_data):
            if not self.running:
                break
            
            batch_start = time.time()
            
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward con mixed precision
            if self.scaler:
                with autocast('cuda'):
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
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            epoch_loss += loss.item()
            n_batches += 1
            self.global_step += 1
            
            # Log periódico (mínimo)
            if self.global_step % self.log_every == 0:
                avg_loss = epoch_loss / n_batches
                speed = input_ids.shape[0] / (sum(batch_times[-10:]) / min(10, len(batch_times)))
                self._update_status(epoch, batch_idx, avg_loss, speed)
            
            # Checkpoint periódico
            if self.global_step % self.checkpoint_every == 0:
                self._save_checkpoint(epoch, epoch_loss / n_batches)
            
            # Yield control al sistema (para que no monopolice recursos)
            if batch_idx % 10 == 0:
                time.sleep(0.01)  # Micro-pausa para no sobrecargar
        
        return epoch_loss / max(n_batches, 1)
    
    def train(self, epochs: int, resume: bool = True):
        """Loop principal de entrenamiento."""
        if resume:
            self.load_checkpoint()
        
        self.start_time = time.time()
        
        print("\n" + "="*50)
        print("🌙 PAMPAr-Coder - Entrenamiento Background")
        print("="*50)
        print(f"   Device: {self.device}")
        print(f"   Epochs: {epochs}")
        print(f"   Checkpoint cada: {self.checkpoint_every} steps")
        print(f"   Status file: {self.status_file}")
        print("="*50)
        print("\n💡 Podés seguir trabajando. El entrenamiento corre en background.")
        print("   Para ver progreso: cat checkpoints/training_status.json")
        print("   Para parar: Ctrl+C (guarda checkpoint automáticamente)\n")
        
        for epoch in range(epochs):
            if not self.running:
                break
            
            epoch_loss = self.train_epoch(epoch)
            
            if self.running:
                print(f"📊 Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Step: {self.global_step}")
                self._save_checkpoint(epoch, epoch_loss)
        
        # Guardar checkpoint final
        self._save_checkpoint(epochs, epoch_loss)
        print(f"\n✅ Entrenamiento completado o pausado.")
        print(f"   Checkpoint guardado en: {self.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description='Entrenamiento background de PAMPAr-Coder')
    parser.add_argument('--data', type=str, default='data/distillation/distillation_data.jsonl',
                        help='Archivo de datos')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directorio de checkpoints')
    parser.add_argument('--checkpoint-every', type=int, default=500,
                        help='Guardar checkpoint cada N steps')
    parser.add_argument('--no-resume', action='store_true',
                        help='No resumir desde checkpoint')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar tokenizer
    tokenizer_path = Path("data/tokenizer/code_tokenizer.model")
    if not tokenizer_path.exists():
        print("❌ Tokenizer no encontrado. Ejecuta primero:")
        print("   python scripts/prepare_tokenizer.py")
        return
    
    import sentencepiece as spm
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(tokenizer_path))
    
    # Cargar datos
    data_path = Path(args.data)
    if not data_path.exists():
        # Intentar con datos existentes
        alt_paths = [
            Path("data/code/train.jsonl"),
            Path("data/distillation/distillation_data.jsonl"),
        ]
        for alt in alt_paths:
            if alt.exists():
                data_path = alt
                break
        else:
            print(f"❌ Datos no encontrados: {data_path}")
            print("   Ejecuta primero: python scripts/download_distillation_data.py")
            return
    
    # Crear dataset y dataloader
    dataset = CodeDataset(data_path, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Crear modelo
    model = crear_modelo("4GB")
    model.config.vocab_size = tokenizer.vocab_size()
    
    # Si hay checkpoint de modelo previo, cargar pesos
    prev_ckpt = Path("checkpoints/last.pt")
    if prev_ckpt.exists():
        print(f"📂 Cargando pesos previos de {prev_ckpt}")
        ckpt = torch.load(prev_ckpt, map_location=device, weights_only=False)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=False)
    
    # Crear trainer
    trainer = BackgroundTrainer(
        model=model,
        train_data=dataloader,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        lr=args.lr,
        checkpoint_every=args.checkpoint_every,
    )
    
    # Entrenar
    trainer.train(args.epochs, resume=not args.no_resume)


if __name__ == "__main__":
    main()
