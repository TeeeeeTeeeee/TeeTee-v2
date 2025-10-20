#!/bin/bash

# INFT System Startup Script
# This script starts both the backend oracle service and the frontend

echo "╔════════════════════════════════════════════════════════════╗"
echo "║       🚀 Starting INFT System                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if backend dependencies are installed
if [ ! -d "backend/node_modules" ]; then
  echo "📦 Installing backend dependencies..."
  cd backend && npm install && cd ..
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
  echo "📦 Installing frontend dependencies..."
  cd frontend && npm install && cd ..
fi

# Check if backend .env exists
if [ ! -f "backend/.env" ]; then
  echo "⚠️  Warning: backend/.env not found!"
  echo "   Creating from .env.example..."
  cp backend/.env.example backend/.env
  echo "   ⚠️  Please edit backend/.env and add your REDPILL_API_KEY"
  echo "   Press Enter to continue or Ctrl+C to cancel..."
  read
fi

# Check if frontend .env.local exists
if [ ! -f "frontend/.env.local" ]; then
  echo "📝 Creating frontend/.env.local..."
  echo "NEXT_PUBLIC_BACKEND_URL=http://localhost:3001" > frontend/.env.local
fi

echo ""
echo "✅ Starting backend on port 3001..."
echo "✅ Starting frontend on port 3000..."
echo ""
echo "📡 Access the INFT UI at: http://localhost:3000/inft"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Start both services using trap to kill all on exit
trap 'kill 0' EXIT

# Start backend in background
cd backend && npm start &
BACKEND_PID=$!

# Give backend time to start
sleep 3

# Start frontend in background
cd ../frontend && npm run dev &
FRONTEND_PID=$!

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID


