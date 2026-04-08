#!/bin/bash
# setup.sh — Install deps and run all tests, then start server

set -e

echo "========================================"
echo "  PaperReviewEnv — Setup & Verify"
echo "========================================"

echo ""
echo "1. Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo "2. Running tests..."
python tests/test_env.py

echo ""
echo "3. Running baseline inference (easy)..."
python inference.py --difficulty easy

echo ""
echo "========================================"
echo "  All checks passed! Starting server..."
echo "  Docs available at: http://localhost:7860/docs"
echo "========================================"
echo ""

uvicorn server.app:app --host 0.0.0.0 --port 7860
