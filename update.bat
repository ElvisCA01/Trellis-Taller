:: IMPORTANT: Save this file as UTF-8 (without BOM) or ANSI!

@echo off
:: ---------------------------------------------------------------------------
:: Re-initializes or updates the Git repository in "code" using the portable Git.
:: Ensures that PATH references the portable Git DLL folders before system paths.
:: ---------------------------------------------------------------------------

:: Optional: set console code page to UTF-8 (if you handle UTF-8 text).
chcp 65001 > nul

:: 1. Define the base folder where MinGit is unpacked.
set PORTABLE_GIT_BASE=%~dp0tools\git

:: 2. Backup the old PATH, then prepend the portable Git paths so they're used first.
set OLD_PATH=%PATH%
set PATH=%PORTABLE_GIT_BASE%\mingw64\bin;%PORTABLE_GIT_BASE%\cmd;%PORTABLE_GIT_BASE%\usr\bin;%PORTABLE_GIT_BASE%\mingw64\libexec\git-core;%PATH%

:: 3. Set Git’s SSL certificate environment variable(s).
::    GIT_SSL_CAINFO has precedence over SSL_CERT_FILE.
set GIT_SSL_CAINFO=%PORTABLE_GIT_BASE%\mingw64\etc\ssl\certs\ca-bundle.crt
set SSL_CERT_FILE=%PORTABLE_GIT_BASE%\mingw64\etc\ssl\certs\ca-bundle.crt

:: Store original directory
set ORIGINAL_DIR=%CD%

echo Starting to update

:: Optional cleanup of venv or other files:
if exist "%~dp0code\venv" (
    echo Removing existing virtual environment...
    rd /s /q "%~dp0code\venv"
)
if exist "%~dp0code\trellis_init_done.txt" (
    echo Removing initialization flag file...
    del "%~dp0code\trellis_init_done.txt"
)

:: Go into "code" folder
cd /d "%~dp0code"

:: ---------------------------------------------------------------------------
:: Check if .git exists; update or init as needed.
:: ---------------------------------------------------------------------------
if exist ".git" (
    echo -----------------------------------------------------------------------------
    echo [INFO] A .git folder already exists. Resetting to remote main branch...
    echo -----------------------------------------------------------------------------

    :: Make sure local config for SSL is correct:
    git config http.sslCAPath "%PORTABLE_GIT_BASE%\mingw64\etc\ssl\certs"
    git config http.sslCAInfo "%PORTABLE_GIT_BASE%\mingw64\etc\ssl\certs\ca-bundle.crt"

    :: 1) Fetch from origin (with minimal history)
    git fetch --depth=1 origin main

    :: 2) Checkout main
    git checkout main

    :: 3) Fetch again to ensure we have the correct HEAD
    git fetch origin main

    :: 4) Reset to FETCH_HEAD
    git reset --hard FETCH_HEAD

    :: 5) Remove any untracked files/directories
    git clean -fd
)

if not exist ".git" (
    echo -----------------------------------------------------------------------------
    echo [INFO] No .git folder found. Initializing a new Git repository...
    echo -----------------------------------------------------------------------------
    git init

    :: Configure SSL certificate path in the new local repo
    git config --local http.sslCAPath "%PORTABLE_GIT_BASE%\mingw64\etc\ssl\certs"
    git config --local http.sslCAInfo "%PORTABLE_GIT_BASE%\mingw64\etc\ssl\certs\ca-bundle.crt"

    :: Add origin and check out
    git remote add origin https://github.com/IgorAherne/trellis-stable-projectorz.git
    git fetch --depth=1 origin main
    git checkout -f main
)

:: ---------------------------------------------------------------------------
:: Initialize and update submodules
:: ---------------------------------------------------------------------------
echo [INFO] Initializing and updating submodules...
git submodule deinit -f --all
git submodule update --init --recursive
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Submodule update failed.
    cd /d "%ORIGINAL_DIR%"
    pause
    exit /b 1
)

echo -----------------------------------------------------------------------------
echo Update Completed.

:: Return to original directory
cd /d "%ORIGINAL_DIR%"
set PATH=%OLD_PATH%
:: Cleanup environment variables if desired:
set GIT_SSL_CAINFO=
set SSL_CERT_FILE=

if not defined SKIP_PAUSE pause
