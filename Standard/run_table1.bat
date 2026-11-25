@echo off
setlocal
if /i not "%~1"=="-Seed" (
  echo Usage: %~nx0 -Seed ^<integer^>
  exit /b 2
)
if "%~2"=="" (
  echo Usage: %~nx0 -Seed ^<integer^>
  exit /b 2
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_table1.ps1" %*
exit /b %ERRORLEVEL%
