import * as vscode from "vscode";
import { ChatPanel } from "./chatPanel";
import { PamparCompletionProvider } from "./completionProvider";
import { InferenceClient } from "./inferenceClient";

let client: InferenceClient | undefined;

export function activate(context: vscode.ExtensionContext): void {
  client = new InferenceClient(context);

  // Comando: abrir chat
  context.subscriptions.push(
    vscode.commands.registerCommand("pampar.openChat", () => {
      ChatPanel.createOrShow(context, client!);
    }),
  );

  // Comando: boot scan del workspace
  context.subscriptions.push(
    vscode.commands.registerCommand("pampar.runBoot", async () => {
      await runBootScan(client!);
    }),
  );

  // Comando: seleccionar checkpoint
  context.subscriptions.push(
    vscode.commands.registerCommand("pampar.setCheckpoint", async () => {
      await selectCheckpoint();
    }),
  );

  // Inline completion provider (Python)
  if (
    vscode.workspace
      .getConfiguration("pampar")
      .get<boolean>("enableInlineCompletion")
  ) {
    const provider = new PamparCompletionProvider(client);
    context.subscriptions.push(
      vscode.languages.registerInlineCompletionItemProvider(
        { language: "python" },
        provider,
      ),
    );
  }

  // WebView en el ActivityBar (sidebar automático)
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      "pamparChat",
      new ChatViewProvider(context, client),
    ),
  );
}

export function deactivate(): void {
  client?.dispose();
}

// ---------------------------------------------------------------------------

async function runBootScan(inferenceClient: InferenceClient): Promise<void> {
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (!workspaceFolders?.length) {
    void vscode.window.showWarningMessage("PAMPAr: No hay workspace abierto.");
    return;
  }
  const root = workspaceFolders[0].uri.fsPath;

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "PAMPAr: Escaneando workspace…",
    },
    async () => {
      const result = await inferenceClient.runBoot(root);
      if (result) {
        void vscode.window.showInformationMessage(
          "PAMPAr: Boot completado. AGENTS.md generado.",
        );
      }
    },
  );
}

async function selectCheckpoint(): Promise<void> {
  const uris = await vscode.window.showOpenDialog({
    filters: { "PyTorch checkpoint": ["pt"] },
    canSelectMany: false,
    title: "Seleccionar checkpoint PAMPAr (.pt)",
  });
  if (!uris?.length) return;
  await vscode.workspace
    .getConfiguration("pampar")
    .update(
      "checkpointPath",
      uris[0].fsPath,
      vscode.ConfigurationTarget.Workspace,
    );
  void vscode.window.showInformationMessage(
    `PAMPAr: Checkpoint → ${uris[0].fsPath}`,
  );
  client?.reload();
}

// ---------------------------------------------------------------------------

/**
 * Proveedor para el WebView en la barra lateral (ActivityBar).
 * Delega en ChatPanel para el HTML y la lógica de mensajes.
 */
class ChatViewProvider implements vscode.WebviewViewProvider {
  constructor(
    private readonly ctx: vscode.ExtensionContext,
    private readonly client: InferenceClient,
  ) {}

  resolveWebviewView(view: vscode.WebviewView): void {
    view.webview.options = { enableScripts: true };
    ChatPanel.attachToView(view, this.ctx, this.client);
  }
}
