@echo off
rem mcnl: program window (Qwen3.5-4B + Micro-Cap 12). Model loads about 40 s after start.
cd /d "%~dp0"
set HF_HUB_OFFLINE=1
if exist "..\venv-hf\Scripts\pythonw.exe" (
  start "" "..\venv-hf\Scripts\pythonw.exe" src\mcnl\gui.py
) else (
  start "" pythonw src\mcnl\gui.py
)
