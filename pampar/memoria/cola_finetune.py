# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Cola de Fine-tune — L3 del sistema de memoria.

Cuando el ClasificadorPareto determina que un fragmento tiene:
  - importancia >= 0.85
  - frecuencia >= 3 veces visto/pedido

...lo promueve a nivel 3 (L3) y se agrega a esta cola.
La cola acumula silenciosamente en disco hasta que:
  A) Tiene suficiente data (umbral configurable, default 500 ejemplos)
  B) El usuario lo pide explícitamente

Cuando se activa el fine-tune:
  1. Se genera un dataset JSONL desde la cola
  2. Se propone al usuario con estadísticas
  3. Si acepta → se lanza el script de training
  4. Si no → la cola se conserva y sigue acumulando
  5. Post-training exitoso → se vacía la cola L3 del RAG
     (el conocimiento ya vive en los pesos)

Analogía: "ya sé caminar, no necesito recordar cómo hacerlo."
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Callable, List, Optional

from .clasificador import EntradaMemoria


class ColaFinetune:
    """
    Gestiona la cola de ejemplos de nivel L3 para fine-tune futuro.

    Args:
        directorio:       Ruta del directorio de datos de memoria
        min_ejemplos:     Mínimo de ejemplos para proponer fine-tune (default 200)
        callback_proponer: Función que se llama cuando la cola está lista
            Signature: (n_ejemplos: int, stats: dict) -> bool
            Si retorna True → se lanza el fine-tune
    """

    def __init__(
        self,
        directorio: str = "memoria/data",
        min_ejemplos: int = 200,
        callback_proponer: Optional[Callable] = None,
    ):
        self.dir = Path(directorio)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ruta_cola = self.dir / "cola_finetune.jsonl"
        self.ruta_log = self.dir / "finetune_log.json"
        self.min_ejemplos = min_ejemplos
        self.callback_proponer = callback_proponer

        self._cola: List[EntradaMemoria] = []
        self._cargar()

    # ── Persistencia ──────────────────────────────────────────────────────────

    def _cargar(self) -> None:
        """Carga cola desde disco."""
        if not self.ruta_cola.exists():
            return
        try:
            for linea in self.ruta_cola.read_text(encoding="utf-8").splitlines():
                if linea.strip():
                    data = json.loads(linea)
                    e = EntradaMemoria(**{
                        k: v for k, v in data.items()
                        if k in EntradaMemoria.__dataclass_fields__
                    })
                    self._cola.append(e)
        except Exception:
            self._cola = []

    def _guardar_cola(self) -> None:
        """Persiste cola en formato JSONL."""
        lineas = [
            json.dumps(
                {k: getattr(e, k) for k in e.__dataclass_fields__},
                ensure_ascii=False,
            )
            for e in self._cola
        ]
        self.ruta_cola.write_text("\n".join(lineas), encoding="utf-8")

    # ── Operaciones ───────────────────────────────────────────────────────────

    def agregar(self, entrada: EntradaMemoria) -> None:
        """
        Agrega una entrada L3 a la cola.

        Si después de agregar la cola supera min_ejemplos,
        llama al callback_proponer si está definido.
        """
        if entrada.nivel < 3:
            return

        # Evitar duplicados
        ids = {e.id for e in self._cola}
        if entrada.id in ids:
            return

        self._cola.append(entrada)
        self._guardar_cola()

        # Proponer fine-tune si tenemos suficientes ejemplos
        if len(self._cola) >= self.min_ejemplos and self.callback_proponer:
            lanzar = self.callback_proponer(len(self._cola), self.stats())
            if lanzar:
                self.lanzar_finetune()

    def __len__(self) -> int:
        return len(self._cola)

    def stats(self) -> dict:
        """Estadísticas de la cola actual."""
        if not self._cola:
            return {
                "total": 0,
                "listos": False,
                "faltan_para_finetune": self.min_ejemplos,
            }

        importancias = [e.importancia for e in self._cola]
        return {
            "total": len(self._cola),
            "listos": len(self._cola) >= self.min_ejemplos,
            "importancia_promedio": round(sum(importancias) / len(importancias), 3),
            "importancia_max": round(max(importancias), 3),
            "territorios": self._contar_territorios(),
            "faltan_para_finetune": max(0, self.min_ejemplos - len(self._cola)),
        }

    def _contar_territorios(self) -> dict:
        cont: dict[str, int] = {}
        for e in self._cola:
            cont[e.territorio_dominante] = cont.get(e.territorio_dominante, 0) + 1
        return cont

    def exportar_dataset(self, ruta_salida: Optional[str] = None) -> Path:
        """
        Exporta la cola como dataset JSONL en formato de instrucción.

        Formato:
            {"instruction": "...", "input": "", "output": "..."}

        Compatible con la mayoría de scripts de fine-tune (Alpaca format).

        Args:
            ruta_salida: Ruta del archivo JSONL. Default: memoria/data/ft_dataset.jsonl
        Returns:
            Path del archivo generado
        """
        ruta = Path(ruta_salida) if ruta_salida else self.dir / "ft_dataset.jsonl"

        lineas = []
        for e in self._cola:
            if e.tipo == "codigo":
                ejemplo = {
                    "instruction": "Completa, refactoriza o explica el siguiente código Python:",
                    "input": e.texto,
                    "output": "",  # Se completará con distilación si está disponible
                }
            elif e.tipo == "error":
                ejemplo = {
                    "instruction": "El siguiente código tiene un error. Identifícalo y corrígelo:",
                    "input": e.texto,
                    "output": "",
                }
            else:
                ejemplo = {
                    "instruction": e.texto,
                    "input": "",
                    "output": "",
                }

            lineas.append(json.dumps(ejemplo, ensure_ascii=False))

        ruta.write_text("\n".join(lineas), encoding="utf-8")
        return ruta

    def lanzar_finetune(
        self,
        script: str = "scripts/train.py",
        checkpoint: str = "checkpoints/pampar_v3_best.pt",
    ) -> bool:
        """
        Exporta el dataset y lanza el script de fine-tune.

        El script recibe el dataset via argumento --data.
        El proceso corre en background — el modelo sigue disponible.

        Args:
            script:     Path al script de training
            checkpoint: Checkpoint base a partir del cual hacer fine-tune
        Returns:
            True si se inició correctamente
        """
        dataset_path = self.exportar_dataset()
        stats = self.stats()

        log_entry = {
            "timestamp": time.time(),
            "n_ejemplos": len(self._cola),
            "stats": stats,
            "dataset": str(dataset_path),
            "status": "iniciado",
        }

        try:
            proc = subprocess.Popen(
                [
                    "python", script,
                    "--data", str(dataset_path),
                    "--checkpoint", checkpoint,
                    "--finetune",
                    "--epochs", "3",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            log_entry["pid"] = proc.pid
            log_entry["status"] = "corriendo"
        except FileNotFoundError:
            log_entry["status"] = "error_script_no_encontrado"
            self._registrar_log(log_entry)
            return False
        except Exception as exc:
            log_entry["status"] = f"error: {exc}"
            self._registrar_log(log_entry)
            return False

        self._registrar_log(log_entry)
        return True

    def vaciar_post_finetune(self) -> int:
        """
        Vacía la cola después de un fine-tune exitoso.

        Llama a esto cuando el training terminó sin errores.
        Returns: número de ejemplos eliminados
        """
        n = len(self._cola)
        self._cola = []
        if self.ruta_cola.exists():
            self.ruta_cola.unlink()
        self._registrar_log({
            "timestamp": time.time(),
            "evento": "cola_vaciada_post_finetune",
            "ejemplos_eliminados": n,
        })
        return n

    def _registrar_log(self, entry: dict) -> None:
        """Agrega una entrada al log de operaciones de fine-tune."""
        log = []
        if self.ruta_log.exists():
            try:
                log = json.loads(self.ruta_log.read_text(encoding="utf-8"))
            except Exception:
                pass
        log.append(entry)
        self.ruta_log.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    def proponer_usuario(self) -> str:
        """
        Genera mensaje legible para mostrar al usuario.

        El runtime/agente.py llama esto cuando la cola está lista.
        """
        s = self.stats()
        if not s["listos"]:
            return (
                f"[Memoria L3] {s['total']}/{self.min_ejemplos} ejemplos acumulados. "
                f"Faltan {s['faltan_para_finetune']} para proponer entrenamiento."
            )

        territorios_str = ", ".join(
            f"{k}: {v}" for k, v in s.get("territorios", {}).items()
        )
        return (
            f"\n[Propuesta de aprendizaje]\n"
            f"Tengo {s['total']} patrones importantes que aprendí de tus interacciones.\n"
            f"Territorios: {territorios_str}\n"
            f"Importancia promedio: {s['importancia_promedio']}\n\n"
            f"¿Querés que me entrene con esto para interiorizar ese conocimiento?\n"
            f"(Si aceptás, el fine-tune corre en background y la memoria residual se libera.)\n"
            f"Respondé: sí / no"
        )
