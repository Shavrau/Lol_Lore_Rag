@echo off
title Runeterra Lore RAG - Servidor Local
echo ========================================================
echo   Iniciando Runeterra Lore RAG (Servidor Local)
echo ========================================================
echo.
echo Abrindo em: http://127.0.0.1:8080
echo Pressione CTRL+C nesta janela para encerrar.
echo.

start "" http://127.0.0.1:8080
uv run python src/server.py

pause
