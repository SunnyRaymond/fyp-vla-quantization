@echo off
setlocal
title NSCC ASPIRE2A

set "ROOT_DIR=%~dp0"
set "PYTHON_EXE=%ROOT_DIR%nscc-access\.venv\Scripts\python.exe"
set "CONNECT_SCRIPT=%ROOT_DIR%nscc-access\aspire2a_shell.py"

if not exist "%PYTHON_EXE%" (
    echo The prepared Python environment was not found:
    echo %PYTHON_EXE%
    echo.
    echo See ASPIRE2A_README.md for setup and troubleshooting.
    pause
    exit /b 1
)

if not exist "%CONNECT_SCRIPT%" (
    echo The ASPIRE2A connection script was not found:
    echo %CONNECT_SCRIPT%
    pause
    exit /b 1
)

"%PYTHON_EXE%" "%CONNECT_SCRIPT%" %*
set "CONNECT_EXIT=%ERRORLEVEL%"

if /I "%~1"=="--check" exit /b %CONNECT_EXIT%

echo.
if not "%CONNECT_EXIT%"=="0" echo Connection ended with an error. See the message above.
echo Press any key to close this window.
pause >nul
exit /b %CONNECT_EXIT%
