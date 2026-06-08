@echo off
rem ── Doppelklick-Starter fuer Mesh -> Ellipsoid SDF ──
rem Wechselt ins Projektverzeichnis und startet die App MIT Konsole,
rem damit alle Ausgaben (Prints, Tracebacks, Library-Banner) sichtbar sind.
cd /d "%~dp0"
".venv\Scripts\python.exe" "main.py"
