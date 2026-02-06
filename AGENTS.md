# PAMPAr-Coder Repository Guidelines

> For AI agents: Claude Code, Codex (OpenAI), Gemini CLI, etc.
> For GitHub Copilot: see `.github/copilot-instructions.md`

## Quick Reference

| Area | Convention |
|------|------------|
| Language | Python 3.11+ |
| Framework | PyTorch 2.x |
| Tokenizer | SentencePiece BPE |
| Testing | pytest |
| Type hints | Always required |
| Docstrings | Google style |
| Cloud | RunPod (A40/A100) |

## Project Structure

```
PAMPAr-Coder/
├── pampar/
│   └── coder/
│       ├── model_v2.py        # Main model (52 Brodmann zones)
│       ├── talamo_v2.py       # Thalamus orchestrator + LLAVES
│       └── zonas_brodmann.py  # Zone definitions
├── cloud/
│   └── runpod/
│       ├── train_cloud.py     # Cloud training script
│       └── config_3b.py       # Model configurations
├── data/
│   ├── tokenizer/             # BPE models
│   └── distillation/          # Training data (JSONL)
├── checkpoints/               # Saved models
├── scripts/                   # Local training/eval
└── tests/                     # pytest tests
```

## Architecture Overview

PAMPAr-Coder uses a brain-inspired architecture:

1. **52 Brodmann Zones** - Specialized code processing areas
2. **4 Territories** - SINTAXIS, SEMÁNTICA, LÓGICO, ESTRUCTURAL
3. **LLAVES** - Rule-based token classification (80% weight)
4. **Thalamus** - Central orchestrator routing tokens to zones

## Critical Rules

- LLAVES are regex patterns, NOT learned - never train them
- Territories process in parallel - keep independent
- INT4 quantization ONLY for LLAVES lookup tables
- Model weights stay in FP16/BF16
- Always use gradient checkpointing for models >500M params
- vocab_size MUST match tokenizer.GetPieceSize()

## Naming Conventions

- **Spanish** for domain concepts: `Talamo`, `Territorio`, `Zona`, `LLAVES`
- **English** for standard ML: `forward`, `embedding`, `hidden_states`
- **Config classes**: `ConfigPampaRCoderV2`, `Config3B`, `Config1_5B`

## Common Tasks

### Train locally
```bash
python scripts/train.py --config 1.5B --epochs 10
```

### Train on RunPod
```bash
ssh root@IP -p PORT
cd /workspace/PAMPAr-Coder
nohup python3 cloud/runpod/train_cloud.py --config 3B > training.log 2>&1 &
```

### Run tests
```bash
pytest tests/ -v
```

## Instructions Files

Detailed instructions in `.github/instructions/`:
- `pampar-architecture.instructions.md` - Brodmann zones, LLAVES, territories
- `cloud-training.instructions.md` - RunPod training guide
