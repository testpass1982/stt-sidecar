#Requires -Version 5.1
<#
.SYNOPSIS
    STT Sidecar — запуск локального VOSK STT сервера на Windows.
#>
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Проверка Python
$py = Get-Command "python" -ErrorAction SilentlyContinue
if (-not $py) {
    $py = Get-Command "python3" -ErrorAction SilentlyContinue
}
if (-not $py) {
    Write-Host "❌ Python не найден. Установите Python 3.10+ с https://python.org" -ForegroundColor Red
    exit 1
}

# Виртуальное окружение
$venv = Join-Path $ScriptDir ".venv"
if (-not (Test-Path $venv)) {
    Write-Host "📦 Создаю виртуальное окружение..." -ForegroundColor Cyan
    & $py.Source -m venv $venv
}

# Активация
$pip = Join-Path $venv "Scripts\pip.exe"
$python = Join-Path $venv "Scripts\python.exe"

# Установка зависимостей
Write-Host "📦 Устанавливаю зависимости..." -ForegroundColor Cyan
& $pip install -q -r (Join-Path $ScriptDir "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Ошибка установки зависимостей" -ForegroundColor Red
    exit 1
}

# Запуск
Write-Host "🎤 Запуск STT Sidecar..." -ForegroundColor Cyan
& $python (Join-Path $ScriptDir "tui_app.py")
