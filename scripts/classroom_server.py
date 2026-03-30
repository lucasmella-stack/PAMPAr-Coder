"""classroom_server.py — HTTP SSE server y CLI para el Classroom."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import traceback
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from classroom_curriculum import ClassroomConfig


class ClassroomHandler(SimpleHTTPRequestHandler):
    """Handler HTTP con SSE para la UI del classroom."""

    engine: object = None  # ClassroomEngine (lazy import para evitar circular)
    ui_path: str = ""

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self._serve_ui()
        elif self.path == "/events":
            self._serve_sse()
        elif self.path == "/status":
            self._serve_status()
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/start":
            self._handle_start()
        elif self.path == "/stop":
            self._handle_stop()
        elif self.path == "/save":
            self._handle_save()
        else:
            self.send_error(404)

    def _serve_ui(self) -> None:
        """Sirve el archivo HTML de la UI."""
        try:
            ui_file = Path(self.ui_path)
            content = ui_file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, str(e))

    def _serve_sse(self) -> None:
        """Server-Sent Events stream."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            while True:
                try:
                    event = self.engine.event_queue.get(timeout=1.0)
                    msg = f"event: {event['event']}\ndata: {event['data']}\n\n"
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_status(self) -> None:
        """Estado actual del aula."""
        e = self.engine
        status = {
            "lesson_count": e.lesson_count,
            "level": e.current_level,
            "accuracy": e.total_correct / max(1, e.lesson_count),
            "replay_size": len(e.replay),
            "session_log_size": len(e.session_log),
        }
        body = json.dumps(status).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_start(self) -> None:
        """Inicia las lecciones en background."""
        threading.Thread(target=self._run_lessons, daemon=True).start()
        self._json_response({"status": "started"})

    def _handle_stop(self) -> None:
        """Detiene las lecciones."""
        self.engine._running = False
        self._json_response({"status": "stopped"})

    def _handle_save(self) -> None:
        """Guarda la sesión."""
        path = self.engine.save_session()
        rec_path = self.engine.save_recording()
        self.engine._save_checkpoint()
        self._json_response({"status": "saved", "path": path, "recording": rec_path})

    def _run_lessons(self) -> None:
        """Loop principal de lecciones."""
        self.engine._running = True
        try:
            while (
                self.engine._running
                and self.engine.lesson_count < self.engine.config.max_lessons
            ):
                self.engine.run_lesson()
                time.sleep(1)
        except Exception as e:
            self.engine._emit("error", f"Error: {e}\n{traceback.format_exc()}")
        finally:
            self.engine._emit("system", "Sesión finalizada.")
            self.engine.save_session()
            self.engine.save_recording()
            self.engine._save_checkpoint()

    def _json_response(self, data: dict, code: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args) -> None:
        """Silenciar logs del HTTP server."""
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="PAMPAr Classroom — Aula simulada")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/v3_ghidra_v9.pt",
        help="Checkpoint del alumno",
    )
    parser.add_argument(
        "--checkpoint-out",
        default="checkpoints/v3_classroom.pt",
        help="Donde guardar el progreso",
    )
    parser.add_argument(
        "--teacher",
        choices=["github", "openrouter", "qwen"],
        default="qwen",
        help="Backend del profesor",
    )
    parser.add_argument("--model", default="qwen-plus", help="Modelo del profesor")
    parser.add_argument(
        "--api-key",
        default="",
        help="API key (o usa GITHUB_TOKEN / OPENROUTER_API_KEY / QWEN_API_KEY)",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate base")
    parser.add_argument("--ewc-lambda", type=float, default=50.0, help="Fuerza EWC")
    parser.add_argument(
        "--max-lessons", type=int, default=200, help="Máximo de lecciones"
    )
    parser.add_argument(
        "--port", type=int, default=8888, help="Puerto del servidor web"
    )
    parser.add_argument("--no-ui", action="store_true", help="Solo consola, sin UI web")
    parser.add_argument(
        "--no-bio", action="store_true", help="Desactivar mecanismos bio-inspirados"
    )
    parser.add_argument("--level", type=int, default=1, help="Nivel inicial (1-5)")

    args = parser.parse_args()

    config = ClassroomConfig(
        checkpoint_in=args.checkpoint,
        checkpoint_out=args.checkpoint_out,
        teacher_backend=args.teacher,
        teacher_model=args.model,
        api_key=args.api_key,
        lr_base=args.lr,
        ewc_lambda=args.ewc_lambda,
        max_lessons=args.max_lessons,
        port=args.port,
        start_level=args.level,
        bio_enabled=not args.no_bio,
    )

    # Import engine aquí (evita circular imports)
    from classroom import ClassroomEngine

    engine = ClassroomEngine(config)
    engine.load()

    if not engine.teacher:
        print("\n❌ No se pudo configurar el profesor. Revisa tu API key.")
        sys.exit(1)

    if args.no_ui:
        print("\n" + "=" * 60)
        print("  🏫 PAMPAr CLASSROOM — Modo consola")
        print("=" * 60)
        engine._running = True
        try:
            while engine._running and engine.lesson_count < config.max_lessons:
                engine.run_lesson()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n\n  Interrumpido por usuario.")
        finally:
            engine.save_session()
            rec = engine.save_recording()
            engine._save_checkpoint()
            accuracy = engine.total_correct / max(1, engine.lesson_count)
            print(
                f"\n  📊 Resumen: {engine.lesson_count} lecciones, {accuracy:.1%} accuracy, nivel {engine.current_level}"
            )
            if rec:
                print(f"  🎥 Grabación guardada: {rec}")
    else:
        ui_path = Path(__file__).parent / "classroom.html"
        ClassroomHandler.engine = engine
        ClassroomHandler.ui_path = str(ui_path)

        server = HTTPServer(("127.0.0.1", config.port), ClassroomHandler)
        print(f"\n  🏫 PAMPAr CLASSROOM — UI en http://localhost:{config.port}")
        print(f"  Presiona Ctrl+C para detener\n")

        try:
            import webbrowser

            webbrowser.open(f"http://localhost:{config.port}")
        except Exception:
            pass

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n  Detenido.")
            engine.save_session()
            engine.save_recording()
            engine._save_checkpoint()
            server.server_close()


if __name__ == "__main__":
    main()
