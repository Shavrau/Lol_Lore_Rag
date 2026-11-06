@echo off
title Runeterra Lore RAG - Web Server & Public Tunnel
echo ========================================================
echo   Iniciando Runeterra Lore RAG ^& Cloudflare Tunnel...
echo ========================================================

start "" uv run python src/server.py
timeout /t 4 /nobreak >nul

echo.
echo Iniciando tunel publico seguro...
start "" cloudflared.exe tunnel --url http://127.0.0.1:8080

echo.
echo Servidor iniciado localmente em: http://127.0.0.1:8080
echo Verifique a janela do Cloudflare Tunnel para obter o link publico https://...trycloudflare.com
echo ========================================================
pause
