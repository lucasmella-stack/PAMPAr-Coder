# PAMPAr-Coder Repository Guidelines

> For AI agents: Claude Code, Codex (OpenAI), Gemini CLI, etc.
> For GitHub Copilot: see `.github/copilot-instructions.md`

## Quick Reference

| Area | Convention |
|------|------------|
| Language | Python 3.11+ |
| Framework | PyTorch 2.x |
| Tokenizer | SentencePiece BPE (48K vocab) |
| Testing | pytest |
| Type hints | Always required |
| Docstrings | Google style |
| Cloud | RunPod (RTX A5000 / A40 / A100) |

## Project Structure

```
PAMPAr-Coder/
├── pampar/
│   └── coder/
│       └── v2/
│           ├── modelo.py          # PampaRCoderV2 main model
│           ├── config.py          # ConfigPampaRCoderV2 + presets
│           ├── talamo.py          # TálamoBrodmann orchestrator
│           ├── llaves.py          # LLAVES INT8 lookup tables
│           ├── zonas.py           # 52 Brodmann zone definitions
│           ├── bloques.py         # BloqueTerrritorial + symbiotic FFN
│           └── aprendizaje/       # Learning subsystems
│               ├── metacognicion.py      # Self-monitoring
│               ├── neuroplasticidad.py   # Dynamic pruning/growth
│               ├── memoria_errores.py    # Error memory + interiorization
│               ├── aprendizaje_online.py # Online adaptation
│               ├── self_play.py          # Self-play training
│               ├── curriculum.py         # Curriculum scheduling
│               └── destilacion.py        # Knowledge distillation
├── cloud/
│   └── runpod/
│       ├── train_cloud.py     # Cloud training script
│       └── config_3b.py       # Model configurations
├── data/
│   ├── tokenizer/             # BPE models (pampar_48k.model)
│   └── distillation/          # Training data (JSONL)
├── checkpoints/               # Saved models
├── scripts/                   # Training, evaluation, utilities
├── tests/                     # pytest tests
└── versions/                  # Legacy code archive
```

## Architecture Overview

PAMPAr-Coder uses a brain-inspired architecture:

1. **52 Brodmann Zones** - Specialized code processing areas
2. **4 Territories** - SINTAXIS (1-15), SEMÁNTICA (16-30), LÓGICO (31-42), ESTRUCTURAL (43-52)
3. **LLAVES** - Rule-based token classification (INT8 quantized, 80% weight)
4. **Thalamus** - Central orchestrator with causal context window (Conv1D, 32 tokens)
5. **Symbiotic Relationships** - Territories support each other via bottleneck projections
6. **Early Exit** - Percentile-10 per-token confidence (focuses on hardest tokens)
7. **Error Memory** - Ring buffer with auto-interiorization after 5 consecutive successes

## Critical Rules

- LLAVES are regex patterns, NOT learned - never train them
- Territories process in parallel, then combine via symbiotic support
- INT8 quantization for LLAVES lookup tables (256 levels, <0.4% error)
- Model weights stay in FP16/BF16
- Always use gradient checkpointing for models >500M params
- vocab_size MUST match tokenizer.GetPieceSize() (48000)
- Context window (32 tokens) uses causal convolution - pad left only

## Naming Conventions

- **Spanish** for domain concepts: `Talamo`, `Territorio`, `Zona`, `LLAVES`, `MemoriaErrores`
- **English** for standard ML: `forward`, `embedding`, `hidden_states`
- **Config classes**: `ConfigPampaRCoderV2`
- **Presets**: `PRESET_4GB`, `PRESET_1_5B`, `PRESET_3B`

## Common Tasks

### Train locally
```bash
python scripts/train.py --config 1.5B --epochs 10
```

### Train on RunPod
```bash
ssh root@IP -p PORT
cd /workspace/PAMPAr-Coder
screen -S train
python3 cloud/runpod/train_cloud.py --config 1_5B > training.log 2>&1
```

### Run tests
```bash
pytest tests/ -v
```

## Instructions Files

Detailed instructions in `.github/instructions/`:
- `pampar-architecture.instructions.md` - Brodmann zones, LLAVES, territories
- `cloud-training.instructions.md` - RunPod training guide
