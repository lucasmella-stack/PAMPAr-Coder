import * as vscode from "vscode";
import { InferenceClient } from "./inferenceClient";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

/**
 * Panel de chat WebView de PAMPAr.
 * Puede usarse como panel flotante independiente (createOrShow)
 * o anclado a la vista de la barra lateral (attachToView).
 */
export class ChatPanel {
  private static floatingPanel: vscode.WebviewPanel | undefined;
  private readonly history: ChatMessage[] = [];

  // ------------------------------------------------------------------
  // Panel flotante
  // ------------------------------------------------------------------

  static createOrShow(
    ctx: vscode.ExtensionContext,
    client: InferenceClient,
  ): void {
    if (ChatPanel.floatingPanel) {
      ChatPanel.floatingPanel.reveal();
      return;
    }
    const panel = vscode.window.createWebviewPanel(
      "pamparChat",
      "PAMPAr Chat",
      vscode.ViewColumn.Beside,
      { enableScripts: true, retainContextWhenHidden: true },
    );
    ChatPanel.floatingPanel = panel;
    const instance = new ChatPanel(panel.webview, ctx, client);
    panel.onDidDispose(() => {
      ChatPanel.floatingPanel = undefined;
      instance.dispose();
    });
  }

  // ------------------------------------------------------------------
  // Vista en ActivityBar
  // ------------------------------------------------------------------

  static attachToView(
    view: vscode.WebviewView,
    ctx: vscode.ExtensionContext,
    client: InferenceClient,
  ): void {
    new ChatPanel(view.webview, ctx, client);
  }

  // ------------------------------------------------------------------
  // Instance
  // ------------------------------------------------------------------

  private readonly disposables: vscode.Disposable[] = [];

  private constructor(
    private readonly webview: vscode.Webview,
    private readonly ctx: vscode.ExtensionContext,
    private readonly client: InferenceClient,
  ) {
    webview.html = this.buildHtml();
    this.disposables.push(
      webview.onDidReceiveMessage((msg: { type: string; text?: string }) => {
        void this.handleMessage(msg);
      }),
    );
  }

  dispose(): void {
    for (const d of this.disposables) d.dispose();
  }

  // ------------------------------------------------------------------
  // Message handling
  // ------------------------------------------------------------------

  private async handleMessage(msg: {
    type: string;
    text?: string;
  }): Promise<void> {
    if (msg.type !== "send" || !msg.text?.trim()) return;

    const userText = msg.text.trim();
    this.history.push({ role: "user", content: userText });
    void this.webview.postMessage({ type: "user", text: userText });
    void this.webview.postMessage({ type: "thinking" });

    const cfg = vscode.workspace.getConfiguration("pampar");
    const maxTokens = cfg.get<number>("maxTokens") ?? 256;
    const temperature = cfg.get<number>("temperature") ?? 0.4;

    // Construir prompt con historial (ventana de las últimas 6 entradas)
    const prompt = this.buildPrompt(userText);

    try {
      const response = await this.client.infer({
        prompt,
        max_tokens: maxTokens,
        temperature,
      });

      const assistantText = response.error
        ? `⚠️ Error: ${response.error}`
        : response.text.trim();

      this.history.push({ role: "assistant", content: assistantText });
      void this.webview.postMessage({ type: "assistant", text: assistantText });
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      void this.webview.postMessage({
        type: "assistant",
        text: `⚠️ ${errMsg}`,
      });
    }
  }

  private buildPrompt(latestUserText: string): string {
    // Ventana de contexto: las últimas 3 rondas + la nueva pregunta
    const window = this.history.slice(-6, -1); // excluye el último (ya lo agregamos)
    let ctx = "";
    for (const msg of window) {
      if (msg.role === "user") {
        ctx += `### Problem:\n${msg.content}\n`;
      } else {
        ctx += `### Solution:\n${msg.content}\n`;
      }
    }
    return `${ctx}### Problem:\n${latestUserText}\n### Solution:\n`;
  }

  // ------------------------------------------------------------------
  // HTML
  // ------------------------------------------------------------------

  private buildHtml(): string {
    const nonce = getNonce();
    const csp = `default-src 'none'; script-src 'nonce-${nonce}'; style-src 'unsafe-inline';`;

    return /* html */ `<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy" content="${csp}">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PAMPAr Chat</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      display: flex; flex-direction: column; height: 100vh;
      font-family: var(--vscode-font-family);
      font-size: var(--vscode-font-size);
      color: var(--vscode-foreground);
      background: var(--vscode-editor-background);
    }
    #messages {
      flex: 1; overflow-y: auto; padding: 12px 10px;
      display: flex; flex-direction: column; gap: 10px;
    }
    .bubble {
      max-width: 90%; padding: 8px 12px;
      border-radius: 8px; white-space: pre-wrap; line-height: 1.45;
      word-break: break-word;
    }
    .user {
      align-self: flex-end;
      background: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
    }
    .assistant {
      align-self: flex-start;
      background: var(--vscode-editor-inactiveSelectionBackground);
    }
    .thinking {
      align-self: flex-start; opacity: 0.6; font-style: italic;
    }
    #inputRow {
      display: flex; padding: 8px; border-top: 1px solid var(--vscode-panel-border);
      gap: 6px;
    }
    #input {
      flex: 1; resize: none; height: 60px;
      background: var(--vscode-input-background);
      color: var(--vscode-input-foreground);
      border: 1px solid var(--vscode-input-border, transparent);
      padding: 6px; border-radius: 4px;
      font-family: inherit; font-size: inherit;
    }
    #send {
      align-self: flex-end; padding: 6px 14px;
      background: var(--vscode-button-background);
      color: var(--vscode-button-foreground);
      border: none; border-radius: 4px; cursor: pointer;
    }
    #send:hover { background: var(--vscode-button-hoverBackground); }
  </style>
</head>
<body>
  <div id="messages"></div>
  <div id="inputRow">
    <textarea id="input" placeholder="Escribe tu pregunta…"></textarea>
    <button id="send">Enviar</button>
  </div>

  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const messages = document.getElementById('messages');
    const input = document.getElementById('input');
    const sendBtn = document.getElementById('send');
    let thinkingEl = null;

    function addBubble(cls, text) {
      const div = document.createElement('div');
      div.className = 'bubble ' + cls;
      div.textContent = text;
      messages.appendChild(div);
      messages.scrollTop = messages.scrollHeight;
      return div;
    }

    function removeThinking() {
      if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; }
    }

    sendBtn.addEventListener('click', sendMessage);
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });

    function sendMessage() {
      const text = input.value.trim();
      if (!text) return;
      input.value = '';
      vscode.postMessage({ type: 'send', text });
    }

    window.addEventListener('message', (event) => {
      const msg = event.data;
      if (msg.type === 'user')      { removeThinking(); addBubble('user', msg.text); }
      if (msg.type === 'thinking')  { thinkingEl = addBubble('thinking', '⏳ Pensando…'); }
      if (msg.type === 'assistant') { removeThinking(); addBubble('assistant', msg.text); }
    });
  </script>
</body>
</html>`;
  }
}

function getNonce(): string {
  const chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let nonce = "";
  for (let i = 0; i < 32; i++) {
    nonce += chars[Math.floor(Math.random() * chars.length)];
  }
  return nonce;
}
