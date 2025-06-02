#!/bin/bash
echo "🤖 Starting Seminote ML Development Environment..."
docker-compose up -d --build
echo "✅ Services started! API: http://localhost:8000"
