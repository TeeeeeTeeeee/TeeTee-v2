@echo off
REM INFT System Startup Script for Windows
REM This script starts both the backend oracle service and the frontend

echo ================================================================
echo        Starting INFT System
echo ================================================================
echo.

REM Check if backend dependencies are installed
if not exist "backend\node_modules" (
  echo Installing backend dependencies...
  cd backend
  call npm install
  cd ..
)

REM Check if frontend dependencies are installed
if not exist "frontend\node_modules" (
  echo Installing frontend dependencies...
  cd frontend
  call npm install
  cd ..
)

REM Check if backend .env exists
if not exist "backend\.env" (
  echo WARNING: backend\.env not found!
  echo Creating from .env.example...
  copy backend\.env.example backend\.env
  echo Please edit backend\.env and add your REDPILL_API_KEY
  pause
)

REM Check if frontend .env.local exists
if not exist "frontend\.env.local" (
  echo Creating frontend\.env.local...
  echo NEXT_PUBLIC_BACKEND_URL=http://localhost:3001 > frontend\.env.local
)

echo.
echo Starting backend on port 3001...
echo Starting frontend on port 3000...
echo.
echo Access the INFT UI at: http://localhost:3000/inft
echo.
echo Press Ctrl+C to stop all services
echo.

REM Start backend in new window
start "INFT Backend" cmd /k "cd backend && npm start"

REM Wait a bit for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend in new window
start "INFT Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Both services are starting in separate windows...
echo Close this window or press any key to exit
pause >nul


