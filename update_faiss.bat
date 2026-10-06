@echo off
setlocal
cd /d "%~dp0"

echo Updating the FAISS course index...
uv run python refresh_index.py
if errorlevel 1 (
    echo.
    echo Index update failed.
    pause
    exit /b 1
)

echo.
echo Index update completed.
pause
