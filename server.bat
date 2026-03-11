::This file is used to start backend and frontend services for this project (only for local use)
::Ignore this file if you are using this through official render link.
::To start this file run ".\server.bat" through terminal.

@echo off
setlocal
echo ===================================================
echo     IntelliDocs (RAG Assistant) Local Startup
echo ===================================================
echo.

:: 0. Stop stale backend Python processes (prevents old route sets)
echo [0/3] Stopping stale backend processes...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do @echo Stopping process on port 8000 PID %%P & taskkill /PID %%P /F >nul 2>&1
echo Port 8000 cleanup complete.
echo.

:: 1. Start PostgreSQL if needed
echo [1/3] Checking PostgreSQL service...
sc query "postgresql-x64-18" | find "RUNNING" >nul 2>&1
if errorlevel 1 (
    echo Starting PostgreSQL...
    net start postgresql-x64-18
) else (
    echo PostgreSQL is already running.
)
echo.

:: 2. Start Backend Server
echo [2/3] Starting FastAPI Backend on port 8000...
cd /d "%~dp0"
set PYTHONPATH=.
set VECTOR_BACKEND=faiss
set STORAGE_BACKEND=local
set AUTH_REQUIRED=false
echo Local backend profile:
echo   VECTOR_BACKEND=%VECTOR_BACKEND%
echo   STORAGE_BACKEND=%STORAGE_BACKEND%
echo   AUTH_REQUIRED=%AUTH_REQUIRED%
start "RAG Backend" cmd /k "cd /d ""%~dp0"" && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 /nobreak >nul
echo Backend started in a new window.
echo.

:: 3. Start Frontend Server
echo [3/3] Starting Vite Frontend on port 5173...
start "RAG Frontend" cmd /c "cd frontend && npm run dev"
echo Frontend started in a new window.
echo.

echo ===================================================
echo All services have been launched!
echo - Frontend: http://localhost:5173
echo - Backend API: http://localhost:8000/docs
echo ===================================================
echo.
if not defined CI pause
