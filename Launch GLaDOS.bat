@echo off
cd /d "%~dp0"
start "" ".\voice_bot\venv\Scripts\pythonw.exe" -m voice_bot.gui
