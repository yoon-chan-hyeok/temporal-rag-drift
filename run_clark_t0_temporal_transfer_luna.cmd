@echo off
setlocal
cd /d "%~dp0"
.\.venv\Scripts\python.exe scripts\run_clark_t0_temporal_transfer_luna.py %*
endlocal
