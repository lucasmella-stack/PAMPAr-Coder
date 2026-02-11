#!/usr/bin/env python3
"""
PAMPAr-Coder 1.5B: Training optimizado para 24GB VRAM (RTX A5000/3090).
Usa gradient checkpointing + 8-bit Adam para caber en 24GB.
"""
import os, sys, json, math, time, logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pampar.coder.v2 import crear_modelo, PRESET_1_5B

# ============================================================
# CONFIG
# ============================================================
CFG = {
    "tokenizer": "data/tokenizer/pampar_48k.model",
    "data_dirs": ["data/code", "data/distillation"],
    "max_seq_len": 512,
    "batch_size": 2,
    "grad_accum": 16,
    "lr": 3e-4,
    "min_lr": 3e-5,
    "warmup_steps": 1000,
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "epochs": 3,
    "save_every": 500,
    "eval_every": 250,
    "log_every": 10,
    "checkpoint_dir": "checkpoints_1_5B",
}

# ============================================================
# Dataset
# ============================================================
class CodeDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len):
        self.samples = samples
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        ids = self.tok.encode(text)[:self.max_len]
        if len(ids) < 2:
            ids = [1, 2]
        ids = ids + [0] * (self.max_len - len(ids))
        t = torch.tensor(ids, dtype=torch.long)
        return t[:-1], t[1:]

def load_data(dirs, tokenizer, max_len, val_pct=0.02):
    samples = []
    for d in dirs:
        p = PROJECT_ROOT / d
        if not p.exists():
            continue
        for f in sorted(p.glob("*.jsonl")):
            logging.info(f"  Loading {f.name}...")
            with open(f, 'r', encoding='utf-8') as fh:
                for line in fh:
                    try:
                        obj = json.loads(line.strip())
                        text = obj.get("text") or obj.get("output") or obj.get("response") or ""
                        instr = obj.get("instruction") or obj.get("input") or obj.get("prompt") or ""
                        if instr and text:
                            full = f"### Instrucción:\n{instr}\n\n### Respuesta:\n{text}"
                        elif text:
                            full = text
                        else:
                            continue
                        if len(full) > 50:
                            samples.append(full)
                    except:
                        continue
            logging.info(f"    Total so far: {len(samples):,}")

    import random
    random.seed(42)
    random.shuffle(samples)
    split = int(len(samples) * (1 - val_pct))
    train_ds = CodeDataset(samples[:split], tokenizer, max_len)
    val_ds = CodeDataset(samples[split:], tokenizer, max_len)
    return train_ds, val_ds

# ============================================================
# Training loop
# ============================================================
def train():
    # Logging
    Path(CFG["checkpoint_dir"]).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(f"{CFG['checkpoint_dir']}/train_{ts}.log"),
            logging.StreamHandler()
        ]
    )

    # GPU
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    logging.info(f"GPU: {gpu_name} ({gpu_mem:.0f}GB)")

    # Tokenizer
    tok = spm.SentencePieceProcessor()
    tok.load(str(PROJECT_ROOT / CFG["tokenizer"]))
    logging.info(f"Tokenizer: {tok.get_piece_size()} tokens")

    # Model
    logging.info("Creating 1.5B model...")
    model = crear_modelo(PRESET_1_5B)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Parameters: {n_params:,} ({n_params/1e9:.2f}B)")

    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logging.info("Gradient checkpointing: ENABLED")
    else:
        # Manual gradient checkpointing for custom models
        for module in model.modules():
            if hasattr(module, 'use_checkpoint'):
                module.use_checkpoint = True
        logging.info("Manual gradient checkpointing: SET")

    model = model.to(device)
    model.train()

    # Data
    logging.info("Loading data...")
    train_ds, val_ds = load_data(CFG["data_dirs"], tok, CFG["max_seq_len"])
    logging.info(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"],
                            shuffle=False, num_workers=1, pin_memory=True)

    # Optimizer - try 8-bit Adam first
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit(
            model.parameters(), lr=CFG["lr"],
            weight_decay=CFG["weight_decay"],
            betas=(0.9, 0.95)
        )
        logging.info("Optimizer: Adam8bit (bitsandbytes)")
    except ImportError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=CFG["lr"],
            weight_decay=CFG["weight_decay"],
            betas=(0.9, 0.95),
        )
        logging.info("Optimizer: AdamW")

    # Scheduler
    total_steps = len(train_loader) * CFG["epochs"] // CFG["grad_accum"]
    warmup = CFG["warmup_steps"]

    def lr_schedule(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    logging.info(f"Steps/epoch: {len(train_loader)//CFG['grad_accum']:,}")
    logging.info(f"Total steps: {total_steps:,}")
    logging.info(f"Batch effective: {CFG['batch_size'] * CFG['grad_accum']}")
    logging.info(f"Seq len: {CFG['max_seq_len']}")

    # Resume from checkpoint
    global_step = 0
    best_val_loss = float('inf')
    start_epoch = 0
    resume_path = Path(CFG["checkpoint_dir"]) / "pampar_1_5B_best.pt"
    if resume_path.exists():
        logging.info(f"Resuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
        except Exception as e:
            logging.warning(f"Could not load optimizer/scheduler state: {e}")
            logging.info("Optimizer reset — continuing with fresh optimizer state")
        global_step = ckpt.get('step', 0)
        start_epoch = ckpt.get('epoch', 0)
        best_val_loss = ckpt.get('val_loss', float('inf'))
        logging.info(f"Resumed at step {global_step}, epoch {start_epoch}, best_val={best_val_loss:.4f}")
        del ckpt
        torch.cuda.empty_cache()
    else:
        logging.info("No checkpoint found, training from scratch")

    # Training
    accum = 0
    total_loss = 0.0
    start_time = time.time()

    for epoch in range(start_epoch, CFG["epochs"]):
        logging.info(f"\n{'='*60}")
        logging.info(f"EPOCH {epoch+1}/{CFG['epochs']}")
        logging.info(f"{'='*60}")

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=0
                )
                loss = loss / CFG["grad_accum"]

            loss.backward()
            total_loss += loss.item()
            accum += 1

            if accum >= CFG["grad_accum"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                accum = 0

                # Log
                if global_step % CFG["log_every"] == 0:
                    avg_loss = total_loss / CFG["log_every"]
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / elapsed
                    eta = (total_steps - global_step) / max(steps_per_sec, 0.01)
                    mem = torch.cuda.max_memory_allocated() / 1e9
                    logging.info(
                        f"Step {global_step:>6} | Loss {avg_loss:.4f} | "
                        f"LR {lr:.2e} | {steps_per_sec:.1f} steps/s | "
                        f"ETA {eta/3600:.1f}h | GPU {mem:.1f}GB"
                    )
                    total_loss = 0.0

                # Eval
                if global_step % CFG["eval_every"] == 0:
                    val_loss = evaluate(model, val_loader, device)
                    logging.info(f"  >>> Val loss: {val_loss:.4f} (best: {best_val_loss:.4f})")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_ckpt(model, optimizer, scheduler, global_step,
                                  epoch, val_loss, "best")
                    model.train()

                # Save
                if global_step % CFG["save_every"] == 0:
                    save_ckpt(model, optimizer, scheduler, global_step,
                              epoch, best_val_loss, f"step_{global_step}")

        # End-of-epoch eval
        val_loss = evaluate(model, val_loader, device)
        logging.info(f"Epoch {epoch+1} done | Val: {val_loss:.4f} | Best: {best_val_loss:.4f}")
        save_ckpt(model, optimizer, scheduler, global_step, epoch, val_loss,
                  f"epoch_{epoch+1}")

    logging.info(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    save_ckpt(model, optimizer, scheduler, global_step, epoch, best_val_loss, "final")


def evaluate(model, loader, device, max_batches=100):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=0
                )
            total += loss.item()
            count += 1
    return total / max(count, 1)


def save_ckpt(model, optimizer, scheduler, step, epoch, val_loss, name):
    path = Path(CFG["checkpoint_dir"]) / f"pampar_1_5B_{name}.pt"
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
        'epoch': epoch,
        'val_loss': val_loss,
        'config': CFG,
    }, path)
    logging.info(f"  Saved: {path} ({path.stat().st_size/1e6:.0f}MB)")

    # Cleanup: keep only last 3 step checkpoints
    ckpts = sorted(Path(CFG["checkpoint_dir"]).glob("pampar_1_5B_step_*.pt"),
                   key=lambda p: p.stat().st_mtime)
    while len(ckpts) > 3:
        ckpts[0].unlink()
        logging.info(f"  Deleted old: {ckpts[0].name}")
        ckpts = ckpts[1:]


if __name__ == "__main__":
    train()
