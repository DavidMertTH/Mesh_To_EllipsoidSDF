@echo off
rem ── Doppelklick-Starter fuer Mesh -> Ellipsoid SDF ──
rem Wechselt ins Projektverzeichnis und startet die App ohne Konsole.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" "main.py"
