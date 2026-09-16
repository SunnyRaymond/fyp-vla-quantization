@echo off
setlocal
title Open ASPIRE2A in VS Code

set "ROOT_DIR=%~dp0"
set "NSCC_CREDENTIAL_FILE=%ROOT_DIR%credentials.env"
set "SSH_ASKPASS=%ROOT_DIR%nscc-access\nscc-askpass.cmd"
set "SSH_ASKPASS_REQUIRE=force"
set "DISPLAY=1"

if not exist "%NSCC_CREDENTIAL_FILE%" (
    echo Credential file not found:
    echo %NSCC_CREDENTIAL_FILE%
    pause
    exit /b 1
)

where code >nul 2>nul
if errorlevel 1 (
    echo The VS Code command-line launcher was not found.
    echo Open VS Code and select: Remote - SSH: Connect to Host... then ASPIRE2A.
    pause
    exit /b 1
)

code --new-window --remote ssh-remote+ASPIRE2A /home/users/ntu/yguo017
exit /b %ERRORLEVEL%
