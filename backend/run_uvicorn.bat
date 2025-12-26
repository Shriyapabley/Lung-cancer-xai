@echo off
cd /d %~dp0
uvicorn app.main:app --reload --port 8000
