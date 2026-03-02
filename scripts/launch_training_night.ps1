# launch_training_night.ps1
# Prepara la PC para training overnight y lanza el proceso con maxima prioridad.

Write-Host ""
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  MODO TRAINING NOCTURNO - PamparV3" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location "c:\Users\lucas\Documents\Be Web\Lunux-AI\PAMPAr-Coder"

# 1. Matar procesos que consumen recursos
Write-Host "[1/5] Cerrando apps innecesarias..." -ForegroundColor Yellow
@("chrome", "msedge", "firefox", "Discord", "slack", "OneDrive", "Teams") | ForEach-Object {
    $killed = Stop-Process -Name $_ -Force -ErrorAction SilentlyContinue -PassThru
    if ($killed) { Write-Host "     Cerrado: $_" }
}

# 2. Desactivar indexacion de Windows Search (consume disco en background)
Write-Host "[2/5] Deteniendo Windows Search Indexer..." -ForegroundColor Yellow
Stop-Service -Name "WSearch" -Force -ErrorAction SilentlyContinue
Write-Host "     OK"

# 3. Plan de energia: Alto rendimiento
Write-Host "[3/5] Activando plan de energia 'Alto rendimiento'..." -ForegroundColor Yellow
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c 2>$null
Write-Host "     OK"

# 4. Evitar suspension
Write-Host "[4/5] Desactivando suspension..." -ForegroundColor Yellow
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
Write-Host "     OK"

# 5. Lanzar training con prioridad AboveNormal
Write-Host "[5/5] Lanzando training..." -ForegroundColor Yellow
$python = "C:\Users\lucas\AppData\Local\Programs\Python\Python313\python.exe"
$trainArgs = "-X utf8 -u scripts/train_v3.py --checkpoint checkpoints/v3_train.pt --lr 1e-5 --batch-size 2 --seq-len 512 --guardar-cada 500"

$proc = Start-Process -PassThru -NoNewWindow -FilePath $python `
    -ArgumentList $trainArgs `
    -RedirectStandardOutput "scripts/train_v3_log.txt" `
    -RedirectStandardError  "scripts/train_v3_err.txt"

Start-Sleep -Seconds 3
$proc.PriorityClass = "AboveNormal"

Write-Host ""
Write-Host "===================================================" -ForegroundColor Green
Write-Host "  TRAINING CORRIENDO" -ForegroundColor Green
Write-Host "  PID       : $($proc.Id)" -ForegroundColor Green
Write-Host "  Prioridad : AboveNormal" -ForegroundColor Green
Write-Host "  Checkpoint: checkpoints/v3_train.pt" -ForegroundColor Green
Write-Host "  Log       : scripts/train_v3_err.txt" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Para monitorear:" -ForegroundColor DarkGray
Write-Host "  Get-Content scripts/train_v3_err.txt -Wait -Tail 10" -ForegroundColor DarkGray
Write-Host ""
