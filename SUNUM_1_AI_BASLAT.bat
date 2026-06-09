@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ================================================
echo   NeYesem AI Backend baslatiliyor
echo   Model yuklenmesi ~20 saniye surer, bekle.
echo   "Application startup complete" yazinca hazir.
echo   Bu pencereyi KAPATMA (sunum boyunca acik kalsin)
echo ================================================
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
echo.
echo AI durdu. Bir tusa bas...
pause >nul
