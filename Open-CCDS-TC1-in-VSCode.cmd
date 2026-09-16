@echo off
setlocal
title Open CCDS-TC1 in VS Code

set "SSH_HOST=CCDS-TC1"
set "SSH_KEY=%USERPROFILE%\.ssh\id_ed25519_ccds_tc1"
set "REMOTE_HOME=/tc1home/UG/yguo017"

if not exist "%SSH_KEY%" (
    echo CCDS SSH key not found:
    echo %SSH_KEY%
    echo Check that the CCDS key is installed before connecting.
    pause
    exit /b 1
)

where code >nul 2>nul
if errorlevel 1 (
    echo The VS Code command-line launcher was not found.
    echo Open VS Code and select: Remote - SSH: Connect to Host... then %SSH_HOST%.
    pause
    exit /b 1
)

code --new-window --remote ssh-remote+%SSH_HOST% %REMOTE_HOME%
exit /b %ERRORLEVEL%
