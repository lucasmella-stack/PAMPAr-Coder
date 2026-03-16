import { ChildProcess, spawn } from "child_process";
import * as path from "path";
import * as readline from "readline";
import * as vscode from "vscode";

export interface InferenceRequest {
  prompt: string;
  max_tokens?: number;
  temperature?: number;
}

export interface InferenceResponse {
  text: string;
  error?: string;
}

export interface BootResult {
  agents_md: string;
}

/**
 * Puente al proceso Python que corre el modelo PamparV3.
 *
 * Protocolo de comunicación: JSON-lines via stdin/stdout.
 *   → { "type": "infer",   "prompt": "...", "max_tokens": 256, "temperature": 0.4 }
 *   → { "type": "boot",    "workspace": "/ruta" }
 *   ← { "type": "infer_ok",  "text": "..." }
 *   ← { "type": "boot_ok",   "agents_md": "..." }
 *   ← { "type": "error",     "message": "..." }
 */
export class InferenceClient implements vscode.Disposable {
  private proc: ChildProcess | undefined;
  private rl: readline.Interface | undefined;
  private pendingResolve: ((resp: InferenceResponse) => void) | undefined;
  private pendingBootResolve: ((resp: BootResult | null) => void) | undefined;
  private ready = false;
  private outputChannel: vscode.OutputChannel;

  constructor(private readonly ctx: vscode.ExtensionContext) {
    this.outputChannel = vscode.window.createOutputChannel("PAMPAr Inference");
    ctx.subscriptions.push(this.outputChannel);
    void this.start();
  }

  // ------------------------------------------------------------------
  // Public API
  // ------------------------------------------------------------------

  async infer(req: InferenceRequest): Promise<InferenceResponse> {
    if (!this.ready) {
      await this.waitReady();
    }
    return new Promise((resolve) => {
      this.pendingResolve = resolve;
      this.send({ type: "infer", ...req });
    });
  }

  async runBoot(workspace: string): Promise<BootResult | null> {
    if (!this.ready) {
      await this.waitReady();
    }
    return new Promise((resolve) => {
      this.pendingBootResolve = resolve;
      this.send({ type: "boot", workspace });
    });
  }

  reload(): void {
    this.stop();
    void this.start();
  }

  dispose(): void {
    this.stop();
    this.outputChannel.dispose();
  }

  // ------------------------------------------------------------------
  // Internal
  // ------------------------------------------------------------------

  private async start(): Promise<void> {
    const cfg = vscode.workspace.getConfiguration("pampar");
    const pythonPath = cfg.get<string>("pythonPath") ?? "python";
    const checkpointPath = cfg.get<string>("checkpointPath") ?? "";
    const device = cfg.get<string>("device") ?? "auto";

    // Si no hay checkpoint configurado, intentar autodetección
    const resolvedCheckpoint = checkpointPath || this.autoDetectCheckpoint();
    if (!resolvedCheckpoint) {
      this.outputChannel.appendLine(
        '[PAMPAr] No hay checkpoint configurado. Ejecuta "PAMPAr: Seleccionar checkpoint".',
      );
      return;
    }

    const scriptArgs = [
      "-m",
      "pampar.inference",
      "--checkpoint",
      resolvedCheckpoint,
      "--device",
      device,
    ];

    // Determinar cwd: raíz del workspace o directorio del paquete pampar
    const workspaceRoot =
      vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? process.cwd();

    this.outputChannel.appendLine(
      `[PAMPAr] Iniciando: ${pythonPath} ${scriptArgs.join(" ")}`,
    );
    this.outputChannel.appendLine(`[PAMPAr] cwd: ${workspaceRoot}`);

    this.proc = spawn(pythonPath, scriptArgs, {
      cwd: workspaceRoot,
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env },
    });

    this.proc.stderr?.on("data", (data: Buffer) => {
      this.outputChannel.append(`[stderr] ${data.toString()}`);
      // Detectar señal de listo desde Python
      const msg = data.toString();
      if (msg.includes("READY")) {
        this.ready = true;
      }
    });

    this.rl = readline.createInterface({ input: this.proc.stdout! });
    this.rl.on("line", (line: string) => this.handleLine(line));

    this.proc.on("exit", (code) => {
      this.outputChannel.appendLine(
        `[PAMPAr] Proceso terminó con código ${code}`,
      );
      this.ready = false;
      this.proc = undefined;
    });
  }

  private stop(): void {
    this.ready = false;
    this.rl?.close();
    this.rl = undefined;
    this.proc?.kill();
    this.proc = undefined;
  }

  private send(data: Record<string, unknown>): void {
    if (!this.proc?.stdin?.writable) {
      this.outputChannel.appendLine("[PAMPAr] stdin no disponible");
      return;
    }
    const line = JSON.stringify(data) + "\n";
    this.proc.stdin.write(line);
  }

  private handleLine(line: string): void {
    this.outputChannel.appendLine(`[stdout] ${line}`);
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(line) as Record<string, unknown>;
    } catch {
      return;
    }

    const type = parsed["type"] as string | undefined;

    if (type === "infer_ok") {
      this.pendingResolve?.({ text: parsed["text"] as string });
      this.pendingResolve = undefined;
    } else if (type === "boot_ok") {
      this.pendingBootResolve?.({ agents_md: parsed["agents_md"] as string });
      this.pendingBootResolve = undefined;
    } else if (type === "error") {
      const errMsg = (parsed["message"] as string) ?? "Error desconocido";
      this.outputChannel.appendLine(`[PAMPAr] ERROR: ${errMsg}`);
      this.pendingResolve?.({ text: "", error: errMsg });
      this.pendingResolve = undefined;
      this.pendingBootResolve?.(null);
      this.pendingBootResolve = undefined;
    } else if (type === "ready") {
      this.ready = true;
    }
  }

  private waitReady(timeoutMs = 30_000): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ready) {
        resolve();
        return;
      }
      const start = Date.now();
      const interval = setInterval(() => {
        if (this.ready) {
          clearInterval(interval);
          resolve();
        } else if (Date.now() - start > timeoutMs) {
          clearInterval(interval);
          reject(new Error("PAMPAr: timeout esperando que Python arranque"));
        }
      }, 200);
    });
  }

  private autoDetectCheckpoint(): string | undefined {
    const folders = vscode.workspace.workspaceFolders;
    if (!folders?.length) return undefined;
    // Buscamos checkpoints/ en la raíz del workspace
    const candidates = [
      path.join(folders[0].uri.fsPath, "checkpoints", "v3_sft_v8.pt"),
      path.join(folders[0].uri.fsPath, "checkpoints", "stable_best.pt"),
    ];
    const fs = require("fs") as typeof import("fs");
    return candidates.find((p) => fs.existsSync(p));
  }
}
