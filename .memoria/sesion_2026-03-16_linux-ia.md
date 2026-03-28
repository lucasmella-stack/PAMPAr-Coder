# Sesión 2026-03-16 — PAMPAr como "el Linux de la IA"

> Archivo de memoria para el agente. Contexto de la sesión de trabajo.

---

## Corrección crítica del agente

El agente redujo PAMPAr a "un generador de código en formato Problem/Solution".
El usuario corrigió: **PAMPAr es un copiloto local autónomo basado en arquitectura cerebral**.

### Lo que PAMPAr ES (nunca olvidar)

1. **Arquitectura cerebral 2D**: grilla cortical 4 streams × 5 niveles, inspirada en el cerebro humano
   - Tálamo (routing) + LLAVES INT8 (80% reglas + 20% aprendido)
   - 4 streams: SINTAXIS (B01-B15), SEMÁNTICA (B16-B30), LÓGICO (B31-B42), ESTRUCTURAL (B43-B52)
   - Lateral Gates = fibras blancas (comunicación entre streams)
   - GQA 4:1 (8Q / 2KV heads), RoPE, Early Exit
   - 108M params, vocab 48K, max_seq_len 4096

2. **Copiloto local 100% offline**: corre en hardware consumer (GTX 1650, 4GB VRAM), sin cloud, sin APIs, sin telemetría

3. **Agente autónomo con RAG del sistema**:
   - Scanner inspecciona: workspace (ast.parse), paquetes (importlib), servicios (socket), hardware (torch.cuda)
   - BootProtocol: CONCIENCIA.md → Scanner → AGENTS.md → RAG L2/L3
   - La identidad (CONCIENCIA.md) es L3 inmutable, el entorno (AGENTS.md) es L2 mutable
   - "Físico con doctorado" que se especializa según el "laboratorio" donde aterriza

4. **Skills reales**: LectorArchivos (ojos), EjecutorCodigo (manos), con Skill ABC para extensibilidad

5. **Memoria con Ley de Pareto**: RAGResidual + ClasificadorPareto (L0→L3) + ColaFinetune (auto-mejora)

6. **Loop autónomo**: prompt → genera → ejecuta → observa → aprende del error → reintenta

7. **Visión final**: el modelo genera su propio AGENTS.md al aterrizar en un sistema nuevo

---

## Analogía Linux ↔ PAMPAr

| Linux                            | PAMPAr                                                | Estado                          |
| -------------------------------- | ----------------------------------------------------- | ------------------------------- |
| Kernel                           | PamparV3 (108M, grilla cortical 2D)                   | ✅ Construido                   |
| Detección hardware (dmesg, udev) | Scanner (ast.parse, importlib, socket, torch.cuda)    | ✅ Construido                   |
| Init system (systemd)            | BootProtocol (CONCIENCIA → Scanner → AGENTS.md → RAG) | ✅ Construido                   |
| Filesystem                       | RAGResidual + ClasificadorPareto                      | ✅ Construido                   |
| Device drivers                   | Skills (Skill ABC → LectorArchivos, EjecutorCodigo)   | ✅ Base, faltan más             |
| Self-compilation                 | ColaFinetune (auto-SFT)                               | ✅ Wiring hecho, no probado e2e |
| Terminal/Shell                   | ??? (cli.py es parche, Continue no integrado)         | ❌ Falta                        |
| Corre en cualquier hardware      | 4GB VRAM, CPU fallback                                | ✅                              |

---

## Brechas detectadas

### 1. Kernel no probado en producción

- 16/16 eval controlado, pero no probado con prompts reales
- SFT actual: Magicoder-OSS-75K (### Problem / ### Solution)
- **El modelo NO fue entrenado para el formato del Agente** ([LEER:], [EJECUTAR:], historial, RAG context)
- Brecha más crítica

### 2. Pocos drivers (skills)

- Solo 2 skills. Faltan: BuscarSkill, GitSkill, TerminalSkill, TestSkill, EditarSkill

### 3. Sin interfaz real

- cli.py llama a generate() directo, NO al Agente (sin RAG, sin skills, sin memoria)
- Continue necesita HTTP server OpenAI-compatible

### 4. Auto-mejora no probada end-to-end

- ColaFinetune → mini-SFT → reload pesos nunca corrió completo

---

## Estrategia propuesta (3 fases)

### Fase A — El kernel funciona de verdad (AHORA)

1. Entrenar modelo con datos en formato del Agente (system prompt + RAG + acciones + historial)
2. CLI usa el Agente real, no generate() directo

### Fase B — Más drivers, shell funcional

3. 3-4 skills más (buscar, editar, git, tests)
4. HTTP server OpenAI-compatible (Continue)
5. Loop auto-mejora probado end-to-end

### Fase C — Distribución empaquetada

6. pip install pampar-coder
7. Integración Continue nativa
8. Documentación tipo man pages

---

## Ventaja competitiva

- Arquitectura cerebral (no transformer genérico)
- 108M params en 4GB VRAM local
- RAG del sistema como contexto (sabe qué hay en tu máquina)
- Auto-aprendizaje (ColaFinetune)
- Sin cloud, sin telemetría, 100% tuyo
- = Propuesta de valor de Linux vs Windows/macOS en los 90s

---

## Estado del proyecto (Mar 2026)

- **Modelo activo**: PamparV3 — 108.3M params, vocab 48K
- **Mejor checkpoint**: v3_sft_v8.pt — 16/16 eval
- **Tests**: 109+ passing
- **Milestone 1** ✅ — 16/16 eval
- **Milestone 2** ✅ — Runtime loop (chat.py + ColaFinetune + mini-SFT wiring)
- **Milestone 3** ✅ — Protocolo (generador determinista AGENTS.md)
- **Milestone 4** ⏳ — VS Code / Continue integration
- **Milestone 5** ⏳ — Voz TTS

## Archivos clave del proyecto

```
pampar/CONCIENCIA.md          — Identidad invariante (L3)
AGENTS.md                     — Protocolo de despliegue (L2, mutable)
ROADMAP.md                    — Plan de evolución
pampar/coder/v3/modelo.py     — PamparV3 (108M)
pampar/coder/v3/talamo.py     — TalamoInicial (routing cerebral)
pampar/coder/v3/bloques.py    — NivelProfundo, StreamFFN, LateralGate
pampar/coder/v3/llaves.py     — LLAVES INT8 (lookup tables)
pampar/coder/v3/zonas.py      — 52 Zonas de Brodmann
pampar/runtime/agente.py      — Agente (orquestador)
pampar/runtime/scanner.py     — Scanner (inspección del entorno)
pampar/runtime/boot.py        — BootProtocol (secuencia de arranque)
pampar/runtime/generar_agents.py — Generador AGENTS.md
pampar/memoria/rag.py         — RAGResidual (vector store)
pampar/memoria/clasificador.py — ClasificadorPareto (L0-L3)
pampar/memoria/cola_finetune.py — ColaFinetune (auto-SFT)
pampar/skills/base.py         — Skill ABC
pampar/skills/lector_archivos.py — LectorArchivos (ojos)
pampar/skills/ejecutar_codigo.py — EjecutorCodigo (manos)
pampar/cli.py                 — CLI (parche, no usa Agente)
pampar/inference.py           — JSON-lines server (base para HTTP)
checkpoints/v3_sft_v8.pt      — Mejor checkpoint (16/16)
data/tokenizer/pampar_48k.model — Tokenizer activo
```

## Hardware del usuario

- **GPU**: GTX 1650 (4GB VRAM)
- **Python**: 3.13 (C:\Users\lucas\AppData\Local\Programs\Python\Python313\python.exe)
- **torch**: 2.6.0+cu124
- **OS**: Windows
- **.venv en Lunux-AI/.venv**: NO tiene torch — no usar para inferencia
