"""classroom_training.py — Tokenización y paso de entrenamiento del Classroom."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class HasLRConfig(Protocol):
    """Protocolo mínimo para la config de LR."""

    lr_base: float
    lr_llaves_mult: float
    lr_attn_mult: float
    lr_embed_mult: float
    lr_ffn_mult: float


def setup_optimizer(
    model: nn.Module,
    config: HasLRConfig,
) -> tuple[torch.optim.Optimizer, float, list[dict]]:
    """Configura AdamW con Learning Rate diferencial.

    Returns:
        (optimizer, baseline_lr, param_group_info) donde param_group_info
        es una lista de dicts con 'label', 'lr', 'n_params' para logging.
    """
    param_groups: list[dict] = []
    assigned: set[str] = set()

    # Grupo 1: LLAVES / Tálamo — casi congelado
    llaves_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in ["talamo", "llaves", "attn_proj"]):
            if param.requires_grad:
                llaves_params.append(param)
                assigned.add(name)
    if llaves_params:
        param_groups.append({
            "params": llaves_params,
            "lr": config.lr_base * config.lr_llaves_mult,
            "label": "llaves_talamo",
        })

    # Grupo 2: Atención — aprende lento
    attn_params = []
    for name, param in model.named_parameters():
        if name not in assigned and any(
            k in name for k in ["attn", "q_proj", "k_proj", "v_proj", "o_proj"]
        ):
            if param.requires_grad:
                attn_params.append(param)
                assigned.add(name)
    if attn_params:
        param_groups.append({
            "params": attn_params,
            "lr": config.lr_base * config.lr_attn_mult,
            "label": "attention",
        })

    # Grupo 3: Embeddings — aprende lento
    embed_params = []
    for name, param in model.named_parameters():
        if name not in assigned and any(k in name for k in ["tok_emb", "emb"]):
            if param.requires_grad:
                embed_params.append(param)
                assigned.add(name)
    if embed_params:
        param_groups.append({
            "params": embed_params,
            "lr": config.lr_base * config.lr_embed_mult,
            "label": "embeddings",
        })

    # Grupo 4: FFN / StreamFFN / todo lo demás
    ffn_params = []
    for name, param in model.named_parameters():
        if name not in assigned and param.requires_grad:
            ffn_params.append(param)
            assigned.add(name)
    if ffn_params:
        param_groups.append({
            "params": ffn_params,
            "lr": config.lr_base * config.lr_ffn_mult,
            "label": "ffn_generation",
        })

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.01)

    info = []
    for g in param_groups:
        n = sum(p.numel() for p in g["params"])
        info.append({"label": g["label"], "lr": g["lr"], "n_params": n})

    return optimizer, config.lr_base, info


def tokenize_pair(
    tokenizer: object,
    problem: str,
    solution: str,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokeniza un par problema→solución con máscara de loss.

    Returns:
        (input_ids, labels) donde labels tiene -100 en el prompt
        para que el loss solo se compute sobre la solución.
    """
    prompt = f"### Problem:\n{problem}\n### Solution:\n```python\n"
    prompt_ids = tokenizer.Encode(prompt)
    solution_ids = tokenizer.Encode(solution + "\n```")

    all_ids = prompt_ids + solution_ids
    if len(all_ids) > seq_len:
        all_ids = all_ids[:seq_len]
        n_prompt = min(len(prompt_ids), len(all_ids))
    else:
        n_prompt = len(prompt_ids)

    input_ids = torch.tensor(all_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[:n_prompt] = -100

    return input_ids, labels


def tokenize_teaching(
    tokenizer: object,
    text: str,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokeniza contenido de enseñanza donde TODOS los tokens son entrenables.

    Se usa para que el alumno absorba explicaciones y ejemplos del mentor.
    """
    ids = tokenizer.Encode(text)
    if len(ids) > seq_len:
        ids = ids[:seq_len]
    input_ids = torch.tensor(ids, dtype=torch.long)
    labels = input_ids.clone()
    return input_ids, labels


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ewc: object,
    examples: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float, dict]:
    """Un paso de entrenamiento con loss masking.

    Args:
        model: Modelo PamparV3.
        optimizer: Optimizer con LR diferencial.
        ewc: Objeto EWC para penalización.
        examples: lista de (input_ids, labels) donde labels=-100 en prompt.
        device: dispositivo de cómputo.

    Returns: (loss_ce, ewc_penalty, last_info)
    """
    model.train()
    optimizer.zero_grad()

    total_loss = torch.tensor(0.0, device=device)
    total_ce = 0.0
    n = 0
    last_info: dict = {}

    for input_ids, labels in examples:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            labels = labels.unsqueeze(0)
        if input_ids.shape[1] < 3:
            continue

        inp = input_ids[:, :-1]
        tgt = labels[:, 1:]
        logits, _, info = model(inp)
        last_info = info

        loss_ce = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=-100,
        )
        total_loss = total_loss + loss_ce
        total_ce += loss_ce.item()
        n += 1

    if n == 0:
        return 0.0, 0.0, {}

    total_loss = total_loss / n

    ewc_pen = ewc.penalty(model)
    total_loss = total_loss + ewc_pen

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return total_ce / n, ewc_pen.item(), last_info
