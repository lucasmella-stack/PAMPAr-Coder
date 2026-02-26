````instructions
# Python ML/LLM Instructions

> Para desarrollo de modelos de ML/LLM con PyTorch.

## Type Hints (OBLIGATORIO)

```python
from typing import Optional, Tuple, Dict, List, Union, Literal
from torch import Tensor
import torch.nn as nn

def forward(
    self,
    input_ids: Tensor,
    attention_mask: Optional[Tensor] = None,
    labels: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Forward pass del modelo.

    Args:
        input_ids: Token IDs, shape (batch, seq_len).
        attention_mask: Máscara de atención, shape (batch, seq_len).
        labels: Labels para calcular loss, shape (batch, seq_len).

    Returns:
        Tuple de (logits, loss). Loss es None si labels no se proporcionan.
    """
    ...
````

## Docstrings (Google Style)

```python
class PampaRCoderV2(nn.Module):
    """
    Modelo principal PAMPAr-Coder V2 con arquitectura cerebral.

    Attributes:
        config: Configuración del modelo.
        embedding: Capa de embedding de tokens.
        talamo: Orquestador central TálamoBrodmann.
        territorios: Lista de 4 BloqueTerrritorial.

    Example:
        >>> config = ConfigPampaRCoderV2.from_preset("1.5B")
        >>> model = PampaRCoderV2(config)
        >>> output = model(input_ids)
    """
```

## PyTorch Patterns

### Model Definition

```python
class MiModulo(nn.Module):
    def __init__(self, config: ConfigPampaRCoderV2):
        super().__init__()
        self.config = config
        # Inicializar layers aquí

    def forward(self, x: Tensor) -> Tensor:
        # Forward pass
        return x

    def _init_weights(self, module: nn.Module) -> None:
        """Inicialización de pesos."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
```

### Training Loop

```python
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    accumulation_steps: int = 4,
) -> float:
    """Entrena una época completa."""
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(tqdm(dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast(dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(dataloader)
```

### Checkpoint Saving/Loading

```python
def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    loss: float,
    path: str,
) -> None:
    """Guarda checkpoint completo."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }, path)

def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
) -> Dict:
    """Carga checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint
```

## Memory Optimization

### Gradient Checkpointing

```python
# Para modelos grandes (>500M params)
model.gradient_checkpointing_enable()

# Manual control:
from torch.utils.checkpoint import checkpoint

class Block(nn.Module):
    def forward(self, x):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)
```

### Efficient Attention

```python
# Usar Flash Attention cuando sea posible
from torch.nn.functional import scaled_dot_product_attention

# O xformers para backwards compatibility
try:
    from xformers.ops import memory_efficient_attention
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False
```

### Tensor Operations

```python
# BIEN: operaciones in-place cuando sea seguro
x.add_(bias)  # En lugar de x = x + bias

# BIEN: evitar concatenaciones innecesarias
# MAL:
# outputs = []
# for block in self.blocks:
#     outputs.append(block(x))
# return torch.cat(outputs, dim=-1)

# BIEN: usar stack si las dimensiones son iguales
outputs = torch.stack([block(x) for block in self.blocks], dim=0)
```

## Testing (pytest)

```python
import pytest
import torch
from pampar.coder.v2.modelo import PampaRCoderV2
from pampar.coder.v2.config import ConfigPampaRCoderV2

@pytest.fixture
def config():
    """Configuración pequeña para tests."""
    return ConfigPampaRCoderV2(
        vocab_size=1000,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
    )

@pytest.fixture
def model(config):
    """Modelo pequeño para tests."""
    return PampaRCoderV2(config)

class TestPampaRCoderV2:
    def test_forward_shape(self, model, config):
        """Verifica output shape."""
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        output = model(input_ids)

        assert output.logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_gradient_flow(self, model):
        """Verifica que gradientes fluyen correctamente."""
        input_ids = torch.randint(0, 1000, (1, 8))
        labels = input_ids.clone()

        output = model(input_ids, labels=labels)
        output.loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
```

## Logging

```python
import logging

# Configurar al inicio del script
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Usar en el código
logger.info(f"Epoch {epoch}: loss={loss:.4f}")
logger.warning(f"GPU memory high: {memory_used:.1f}GB")
logger.error(f"Checkpoint save failed: {e}")
```

```

```
