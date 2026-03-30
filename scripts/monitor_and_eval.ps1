# monitor_and_eval.ps1 — Monitorea el training y lanza evaluación al terminar
$ErrorActionPreference = "Continue"
$projectDir = "C:\Users\lucas\Documents\Be Web\Lunux-AI\PAMPAr-Coder"
$python = "C:\Users\lucas\AppData\Local\Programs\Python\Python313\python.exe"
$targetPid = 8456

Set-Location $projectDir
Write-Host "=== Monitor de Training PAMPAr V3 ===" -ForegroundColor Cyan
Write-Host "Monitoreando PID $targetPid..."
Write-Host ""

# --- Fase 1: Esperar a que termine el training ---
while ($true) {
    $proc = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
    if (-not $proc) {
        Write-Host ""
        Write-Host ">>> Training TERMINADO <<<" -ForegroundColor Green
        break
    }
    
    # Mostrar progreso basado en checkpoints
    $latest = Get-ChildItem "$projectDir\checkpoints\v3_pretrain*" | 
        Sort-Object LastWriteTime -Descending | 
        Select-Object -First 1
    $cpuMin = [math]::Round($proc.CPU / 60, 1)
    $wsMB = [math]::Round($proc.WS / 1MB)
    $now = Get-Date -Format "HH:mm:ss"
    Write-Host "[$now] Training activo | CPU=${cpuMin}min | RAM=${wsMB}MB | Ultimo ckpt: $($latest.Name) ($($latest.LastWriteTime.ToString('HH:mm:ss')))"
    
    Start-Sleep -Seconds 60
}

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  FASE 2: Evaluacion de generacion de codigo" -ForegroundColor Cyan  
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# --- Fase 2: Ejecutar eval_v3.py ---
Write-Host "Ejecutando eval_v3.py..." -ForegroundColor Yellow
& $python -X utf8 scripts/eval_v3.py --checkpoint checkpoints/v3_pretrain_best.pt --verbose 2>&1 | Tee-Object -FilePath "$projectDir\eval_pretrain_results.txt"

Write-Host ""
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  FASE 3: Brain Scanner Suite (GhidraProbe)" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# --- Fase 3: Ejecutar brain_scanner --suite ---
Write-Host "Ejecutando brain_scanner.py --suite..." -ForegroundColor Yellow
& $python -X utf8 scripts/brain_scanner.py --suite --checkpoint checkpoints/v3_pretrain_best.pt 2>&1 | Tee-Object -FilePath "$projectDir\brain_scanner_pretrain_results.txt"

Write-Host ""
Write-Host "=============================================" -ForegroundColor Green
Write-Host "  EVALUACION COMPLETA" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Resultados guardados en:"
Write-Host "  - eval_pretrain_results.txt"
Write-Host "  - brain_scanner_pretrain_results.txt"
