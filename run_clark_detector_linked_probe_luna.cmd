@echo off
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" "scripts\run_clark_detector_linked_probe_luna.py" %*
exit /b %errorlevel%
