import * as vscode from "vscode";
import { InferenceClient } from "./inferenceClient";

/**
 * Proveedor de completado inline (tab completion al estilo Copilot).
 * Se activa solo en archivos Python cuando `pampar.enableInlineCompletion` es true.
 *
 * Lógica de trigger:
 *  - El usuario deja de escribir durante MIN_IDLE_MS ms
 *  - No se dispara si la línea está vacía o es solo un comentario
 *  - Envía hasta CONTEXT_LINES líneas de contexto al modelo
 */
export class PamparCompletionProvider
  implements vscode.InlineCompletionItemProvider
{
  private static readonly MIN_IDLE_MS = 600;
  private static readonly CONTEXT_LINES = 20;
  private static readonly MAX_NEW_TOKENS = 80;

  private lastRequest: number = 0;

  constructor(private readonly client: InferenceClient) {}

  async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    _context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken,
  ): Promise<vscode.InlineCompletionList | null> {
    const cfg = vscode.workspace.getConfiguration("pampar");
    if (!cfg.get<boolean>("enableInlineCompletion")) return null;

    const now = Date.now();
    this.lastRequest = now;

    // Debounce: esperar MIN_IDLE_MS
    await delay(PamparCompletionProvider.MIN_IDLE_MS);
    if (token.isCancellationRequested || this.lastRequest !== now) return null;

    const currentLine = document.lineAt(position.line).text.trimEnd();

    // No completar líneas vacías o solo comentarios
    if (!currentLine || currentLine.trimStart().startsWith("#")) return null;

    const prompt = buildPrompt(
      document,
      position,
      PamparCompletionProvider.CONTEXT_LINES,
    );
    if (!prompt) return null;

    const temp = cfg.get<number>("temperature") ?? 0.3;

    const response = await this.client.infer({
      prompt,
      max_tokens: PamparCompletionProvider.MAX_NEW_TOKENS,
      temperature: temp,
    });

    if (token.isCancellationRequested || response.error) return null;

    const completion = trimCompletion(response.text);
    if (!completion) return null;

    return new vscode.InlineCompletionList([
      new vscode.InlineCompletionItem(
        completion,
        new vscode.Range(position, position),
      ),
    ]);
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function buildPrompt(
  document: vscode.TextDocument,
  position: vscode.Position,
  contextLines: number,
): string {
  const startLine = Math.max(0, position.line - contextLines);
  const lines: string[] = [];
  for (let i = startLine; i <= position.line; i++) {
    const lineText = document.lineAt(i).text;
    lines.push(
      i === position.line ? lineText.slice(0, position.character) : lineText,
    );
  }
  const ctx = lines.join("\n");
  // Formato SFT del modelo: Problem → Solution
  return `### Problem:\nCompleta el siguiente código Python:\n\`\`\`python\n${ctx}\n### Solution:\n\`\`\`python\n${ctx}`;
}

/**
 * Recorta el texto generado para quedarnos solo con la primera "unidad" lógica
 * (hasta la primera línea vacía o el primer bloque completo).
 */
function trimCompletion(raw: string): string {
  // Quitar bloques markdown si el modelo los generó
  const stripped = raw
    .replace(/^```python\n?/i, "")
    .replace(/```$/m, "")
    .trimEnd();

  // Quedarnos con las primeras líneas hasta línea vacía
  const lines = stripped.split("\n");
  const result: string[] = [];
  for (const line of lines) {
    if (line.trim() === "" && result.length > 0) break;
    result.push(line);
  }
  return result.join("\n");
}
