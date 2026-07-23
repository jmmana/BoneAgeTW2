#!/bin/bash
# Levanta backend (FastAPI) y frontend (Vite) en paralelo
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "=== BoneAgeTW2 ==="

# Backend
cd "$ROOT/backend"
if ! command -v uvicorn &>/dev/null; then
  echo "Instalando dependencias Python..."
  pip install -r "$ROOT/requirements.txt" -q
fi
uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID (http://localhost:8000)"

# Frontend
cd "$ROOT/frontend"
if [ ! -d node_modules ]; then
  echo "Instalando dependencias npm..."
  npm install --silent
fi
npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID (http://localhost:5174)"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" EXIT INT TERM
wait
