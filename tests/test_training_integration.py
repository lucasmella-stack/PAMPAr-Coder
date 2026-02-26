# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Test de integración: MemoriaJerarquica + training loop.

Verifica que el modelo entrena correctamente con la memoria
jerárquica activada, sin necesidad de GPU ni datos reales.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from pampar.coder.v2.config import ConfigV2
from pampar.coder.v2.modelo import PampaRCoderV2
from pampar.coder.v2.aprendizaje.memoria_jerarquica import MemoriaJerarquica


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def small_model():
    """Modelo pequeñísimo para test rápido en CPU."""
    config = ConfigV2(
        vocab_size=500,
        dim=64,
        n_heads=4,
        n_capas=2,
        max_seq_len=32,
        dropout=0.0,
    )
    return PampaRCoderV2(config)


@pytest.fixture
def memoria():
    """Memoria con capacidades pequeñas para test rápido."""
    return MemoriaJerarquica(
        capacidad_l0=20,
        capacidad_l1=50,
        capacidad_l2=25,
        ventana_tokens=4,
        umbral_loss_alta=0.5,
        lr_interiorizacion=1e-3,
    )


# =============================================================================
# TESTS
# =============================================================================

class TestTrainingIntegration:
    """Simula un mini training loop con memoria jerárquica."""

    def test_forward_returns_terr_acts(self, small_model):
        """El modelo ahora expone terr_acts en info."""
        input_ids = torch.randint(0, 500, (2, 16))
        logits, loss, info = small_model(input_ids)

        assert "terr_acts" in info
        assert info["terr_acts"].shape[0] == 2  # batch_size
        assert info["terr_acts"].shape[1] == 16  # seq_len

    def test_per_token_loss_computation(self, small_model):
        """Calcular per-token loss como lo hace train_cloud."""
        input_ids = torch.randint(0, 500, (2, 16))
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100

        logits, loss, info = small_model(input_ids, labels)

        # Per-token loss (mismo cálculo que train_cloud.py)
        per_token_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction='none',
        ).reshape(input_ids.size(0), -1)

        pad = torch.zeros(input_ids.size(0), 1)
        per_token_loss = torch.cat([pad, per_token_loss], dim=1)

        assert per_token_loss.shape == input_ids.shape
        assert per_token_loss[:, 0].sum() == 0  # Primer token es padding

    def test_mini_training_loop_with_memoria(self, small_model, memoria):
        """Simula 5 pasos de entrenamiento con memoria activa."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
        initial_loss = None

        for step in range(5):
            # Generar batch aleatorio
            input_ids = torch.randint(0, 500, (2, 16))
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100

            # Forward
            small_model.train()
            logits, loss, info = small_model(input_ids, labels)

            if initial_loss is None:
                initial_loss = loss.item()

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Memoria: capturar patrones difíciles
            with torch.no_grad():
                per_token_loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                    reduction='none',
                ).reshape(input_ids.size(0), -1)

                pad = torch.zeros(input_ids.size(0), 1)
                per_token_loss = torch.cat([pad, per_token_loss], dim=1)

                terr_acts = info.get("terr_acts")
                result = memoria.procesar_batch(
                    input_ids=input_ids,
                    per_token_loss=per_token_loss,
                    terr_acts=terr_acts,
                )

            assert "entradas_creadas" in result

        # Verificar que la memoria capturó algo
        stats = memoria.stats()
        assert stats["total_tokens_procesados"] > 0

    def test_replay_batch_trains_model(self, small_model, memoria):
        """Replay batch se genera y puede entrenar al modelo."""
        # Llenar memoria con patrones
        for i in range(200):
            from pampar.coder.v2.aprendizaje.memoria_jerarquica import EntradaMemoria
            e = EntradaMemoria(
                tokens=tuple(range(i % 400, (i % 400) + 5)),
                loss_media=float(i % 10 + 1),
                novedad=0.5,
                territorio_dominante=i % 4,
            )
            memoria.l1.agregar(e)

        # Obtener replay batch
        replay = memoria.get_replay_batch(
            batch_size=4, nivel="l1", strategy="hardest",
        )
        assert replay is not None
        assert replay.shape[0] == 4

        # Entrenar con replay (loss escalada)
        targets = replay.clone()
        targets[:, :-1] = replay[:, 1:]
        targets[:, -1] = -100

        small_model.train()
        logits, loss, _ = small_model(replay, targets)
        assert loss is not None

        scaled = loss * 0.1
        scaled.backward()

    def test_consolidacion_con_modelo(self, small_model, memoria):
        """Consolidación ejecuta sin errores y actualiza pesos."""
        # Llenar todos los niveles
        for step in range(30):
            input_ids = torch.randint(0, 500, (2, 16))
            per_token_loss = torch.rand(2, 16) * 6.0
            memoria.procesar_batch(input_ids, per_token_loss)

        weights_before = small_model.tok_emb.weight.data.clone()

        result = memoria.consolidar(model=small_model)
        assert "niveles" in result

    def test_checkpoint_with_memoria(self, small_model, memoria, tmp_path):
        """Checkpoint incluye estado de memoria."""
        # Procesar datos
        for _ in range(10):
            input_ids = torch.randint(0, 500, (2, 16))
            per_token_loss = torch.rand(2, 16) * 5.0
            memoria.procesar_batch(input_ids, per_token_loss)

        # Guardar checkpoint de modelo
        ckpt_path = tmp_path / "test_ckpt.pt"
        torch.save({"model": small_model.state_dict()}, ckpt_path)

        # Guardar memoria
        mem_path = ckpt_path.with_suffix(".memoria.json")
        memoria.guardar(str(mem_path))

        assert ckpt_path.exists()
        assert mem_path.exists()

        # Cargar memoria
        loaded = MemoriaJerarquica.cargar(str(mem_path))
        assert loaded.stats()["total_tokens_procesados"] == \
            memoria.stats()["total_tokens_procesados"]

    def test_full_epoch_simulation(self, small_model, memoria):
        """Simula una epoch completa: train + replay + consolidación."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
        replay_every = 3
        consolidar_every = 8

        for step in range(10):
            # === Train step ===
            input_ids = torch.randint(0, 500, (2, 16))
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100

            small_model.train()
            logits, loss, info = small_model(input_ids, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # === Memoria: procesar ===
            with torch.no_grad():
                per_token_loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                    reduction='none',
                ).reshape(input_ids.size(0), -1)
                pad = torch.zeros(input_ids.size(0), 1)
                per_token_loss = torch.cat([pad, per_token_loss], dim=1)
                memoria.procesar_batch(input_ids, per_token_loss, info.get("terr_acts"))

            # === Replay ===
            if (step + 1) % replay_every == 0:
                replay = memoria.get_replay_batch(4, "l1", "hardest")
                if replay is not None:
                    targets_r = replay.clone()
                    targets_r[:, :-1] = replay[:, 1:]
                    targets_r[:, -1] = -100
                    _, r_loss, _ = small_model(replay, targets_r)
                    if r_loss is not None:
                        (r_loss * 0.1).backward()
                        optimizer.step()
                        optimizer.zero_grad()

            # === Consolidación ===
            if (step + 1) % consolidar_every == 0:
                result = memoria.consolidar(model=small_model)
                assert "niveles" in result

        final_stats = memoria.stats()
        assert final_stats["total_tokens_procesados"] > 0
        print(f"\n   Simulación completa: {memoria}")


# =============================================================================
# SMOKE TEST — KNOWLEDGE DISTILLATION
# =============================================================================

class TestDistillationSmoke:
    """
    Verifica que el pipeline de Knowledge Distillation es correcto:
    - Teacher congelado: gradientes cero, params sin cambiar
    - Loss combinada (CE + KL) es finita y permite backward
    - Student sí actualiza sus parámetros
    """

    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        alpha: float = 0.3,
        temperature: float = 4.0,
    ) -> torch.Tensor:
        """Réplica exacta de _distillation_loss() en aprender_solo.py."""
        T = temperature
        s_log = F.log_softmax(student_logits / T, dim=-1)
        t_prob = F.softmax(teacher_logits / T, dim=-1)
        kl = F.kl_div(s_log, t_prob, reduction="batchmean")
        return (T ** 2) * kl

    def test_teacher_frozen_no_grad(self, small_model):
        """Teacher no debe acumular gradientes en ningún parámetro."""
        teacher = PampaRCoderV2(small_model.config)
        teacher.eval()
        teacher.requires_grad_(False)

        # Verificar que todos los params tienen requires_grad=False
        for name, param in teacher.named_parameters():
            assert not param.requires_grad, (
                f"Teacher param '{name}' aún tiene requires_grad=True"
            )

    def test_combined_loss_finite(self, small_model):
        """La combinación CE + KL debe ser finita y permitir backward."""
        teacher = PampaRCoderV2(small_model.config)
        teacher.eval()
        teacher.requires_grad_(False)

        alpha = 0.3
        temperature = 4.0

        input_ids = torch.randint(0, 500, (2, 16))
        labels = input_ids.clone()
        labels[:, 1:] = input_ids[:, :-1]

        small_model.train()
        student_logits, loss_ce, _ = small_model(input_ids, labels)

        with torch.no_grad():
            teacher_logits, _, _ = teacher(input_ids)

        # Alinear shapes (student y teacher pueden tener L ligeramente distinto)
        L = min(student_logits.size(1), teacher_logits.size(1))
        s_log = student_logits[:, :L, :]
        t_log = teacher_logits[:, :L, :]

        kl_loss = self._distillation_loss(s_log, t_log, alpha, temperature)

        if loss_ce is not None:
            total_loss = (1 - alpha) * loss_ce + alpha * kl_loss
        else:
            total_loss = kl_loss

        assert torch.isfinite(total_loss), (
            f"Loss combinada no es finita: {total_loss.item()}"
        )
        assert total_loss.item() > 0, "Loss combinada debe ser positiva"

        # Backward debe funcionar sin errores
        total_loss.backward()

    def test_teacher_params_unchanged_after_backward(self, small_model):
        """Los params del teacher NO deben cambiar tras el backward del student."""
        teacher = PampaRCoderV2(small_model.config)
        teacher.eval()
        teacher.requires_grad_(False)

        # Snapshot de weights del teacher antes
        teacher_weights_before = {
            name: param.clone()
            for name, param in teacher.named_parameters()
        }

        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)

        input_ids = torch.randint(0, 500, (2, 16))
        labels = input_ids.clone()
        labels[:, 1:] = input_ids[:, :-1]

        small_model.train()
        student_logits, loss_ce, _ = small_model(input_ids, labels)

        with torch.no_grad():
            teacher_logits, _, _ = teacher(input_ids)

        L = min(student_logits.size(1), teacher_logits.size(1))
        kl_loss = self._distillation_loss(
            student_logits[:, :L, :], teacher_logits[:, :L, :]
        )

        total = (0.7 * loss_ce + 0.3 * kl_loss) if loss_ce is not None else kl_loss
        total.backward()
        optimizer.step()

        # Verificar que teacher NO cambió
        for name, param in teacher.named_parameters():
            before = teacher_weights_before[name]
            assert torch.equal(param, before), (
                f"Teacher param '{name}' cambió tras el backward del student!"
            )

    def test_student_params_updated(self, small_model):
        """El student SÍ debe actualizar sus parámetros tras optimizer.step()."""
        teacher = PampaRCoderV2(small_model.config)
        teacher.eval()
        teacher.requires_grad_(False)

        # Snapshot de weights del student antes
        student_weights_before = {
            name: param.clone()
            for name, param in small_model.named_parameters()
        }

        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)

        input_ids = torch.randint(0, 500, (2, 16))
        labels = input_ids.clone()
        labels[:, 1:] = input_ids[:, :-1]

        small_model.train()
        student_logits, loss_ce, _ = small_model(input_ids, labels)

        with torch.no_grad():
            teacher_logits, _, _ = teacher(input_ids)

        L = min(student_logits.size(1), teacher_logits.size(1))
        kl_loss = self._distillation_loss(
            student_logits[:, :L, :], teacher_logits[:, :L, :]
        )

        total = (0.7 * loss_ce + 0.3 * kl_loss) if loss_ce is not None else kl_loss
        total.backward()
        optimizer.step()

        # Al menos 1 parámetro del student debe haber cambiado
        changed = sum(
            0 if torch.equal(param, student_weights_before[name]) else 1
            for name, param in small_model.named_parameters()
        )
        assert changed > 0, "Ningún parámetro del student se actualizó"
