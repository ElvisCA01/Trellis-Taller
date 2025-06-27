@echo off
set DIR=%~dp0tools
set PORTABLE_PYTHON=%DIR%\python\python.exe
set BASE_PATH=%DIR%\git\mingw64\bin;%DIR%\python;%DIR%\python\Scripts
set PATH=%BASE_PATH%;%PATH%
set VENV_PATH=%~dp0code\venv

:: Force typical Windows appdata variables if not defined.
:: important to avoid errors with  winreg.QueryValueEx(key, shell_folder_name)
if not defined APPDATA       set APPDATA=%USERPROFILE%\AppData\Roaming
if not defined LOCALAPPDATA  set LOCALAPPDATA=%USERPROFILE%\AppData\Local
if not defined PROGRAMDATA   set PROGRAMDATA=C:\ProgramData

rem Check critical DLLs
if not exist "%DIR%\python\vcruntime140.dll" (
    echo Error: Missing vcruntime140.dll
    exit /b 1
)
if not exist "%DIR%\python\vcruntime140_1.dll" (
    echo Error: Missing vcruntime140_1.dll
    exit /b 1
)
if not exist "%DIR%\python\_ctypes.pyd" (
    echo Error: Missing _ctypes.pyd
    exit /b 1
)
if not exist "%DIR%\python\libffi-8.dll" (
    echo Error: Missing libffi-8.dll
    exit /b 1
)
if not exist "%DIR%\python\libcrypto-3.dll" (
    echo Error: Missing libcrypto-3.dll
    exit /b 1
)
if not exist "%DIR%\python\libssl-3.dll" (
    echo Error: Missing libssl-3.dll
    exit /b 1
)
if not exist "%DIR%\python\_ssl.pyd" (
    echo Error: Missing _ssl.pyd
    exit /b 1
)


:: Prevent Python from using user site packages (APPDATA)
set PYTHONNOUSERSITE=1

rem Clear any existing Python/Conda related env vars
set CONDA_PREFIX=
set CONDA_PYTHON_EXE=
set CONDA_DEFAULT_ENV=
set CONDA_SHLVL=
set CONDAPATH=
set PYTHONPATH=
set PYTHONHOME=


:: First set basic environment variables
set PIP_INSTALLER_LOCATION=%DIR%\python\get-pip.py
set PIP_DISABLE_PIP_VERSION_CHECK=1
::so that pip never tries to read from system config that triggers the registry call:
set PIP_CONFIG_FILE=NUL


echo %DIR%
echo "%PORTABLE_PYTHON%"
echo "%DIR%\python\get-pip.py"

:: create venv if it doesn't exist
if not exist "%VENV_PATH%\Scripts\activate.bat" (

    :: Manually remove potentially broken pip and virtualenv from portable python's site-packages
    echo Cleaning packages from portable Python's site-packages...
    if exist %SITE_PACKAGES%\pip rmdir /s /q %SITE_PACKAGES%\pip
    if exist %SITE_PACKAGES%\pip-*.dist-info del /q %SITE_PACKAGES%\pip-*.dist-info
    if exist %SITE_PACKAGES%\virtualenv rmdir /s /q %SITE_PACKAGES%\virtualenv
    if exist %SITE_PACKAGES%\virtualenv-*.dist-info del /q %SITE_PACKAGES%\virtualenv-*.dist-info
    echo Cleanup finished.

    :: Attempt to install pip using get-pip.py (ensurepip is missing)
    echo Bootstrapping pip using get-pip.py...
    "%PORTABLE_PYTHON%" "%DIR%\python\get-pip.py"
    if errorlevel 1 (
       echo ERROR: get-pip.py failed to install pip in portable Python even after cleanup.
       echo Please check the integrity of the tools\python directory and get-pip.py.
       echo The portable Python distribution might be broken.
       pause
       exit /b 1
    )


    :: Install virtualenv package if missing, using the portable python from THE SPECIFIC filepath:
    :: (avoid the 'python' command during venv creation)
    "%PORTABLE_PYTHON%" -m pip show virtualenv >nul 2>&1
    if errorlevel 1 (
        echo Installing virtualenv using portable Python...
        "%PORTABLE_PYTHON%" -m pip install virtualenv
        if errorlevel 1 (
            echo Failed to install virtualenv using portable Python.
            pause
            exit /b 1
        )
    )

    :: Verify portable python is available before trying to create venv
    "%PORTABLE_PYTHON%" --version >nul 2>&1
    if errorlevel 1 (
        echo Portable Python "%PORTABLE_PYTHON%" not found or failed. Please check the 'tools\python' folder.
        pause
        exit /b 1
    )
    echo Creating virtual environment using portable Python...
    echo "%PORTABLE_PYTHON%" -m virtualenv "%VENV_PATH%"
    "%PORTABLE_PYTHON%" -m virtualenv "%VENV_PATH%"
    if errorlevel 1 (
        echo Failed to create virtual environment using portable Python.
        pause
        exit /b 1
    )
)



:: In environment.bat after venv creation:
:: Always ensure our pip.ini is in place
copy /Y "%DIR%\python\pip.ini" "%VENV_PATH%\pip.ini"
if errorlevel 1 (
    echo Failed to copy pip.ini
    pause
    exit /b 1
)

:: activate venv
call "%VENV_PATH%\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment
    pause
    exit /b 1
)

:: you can set your custom path to python here. Notice, python must be 3.11
:: Don't forget quotation marks ""
:: For example   set PYTHON="C:\Program Files\Python311\python.exe"
set PYTHON="%VENV_PATH%\Scripts\python.exe"

:: If PYTHON is not set, use the activated venv python
if not defined PYTHON (
    set PYTHON=python
)

echo Portable Python located at: %PORTABLE_PYTHON%
echo Virtual environment Python set to: %PYTHON%