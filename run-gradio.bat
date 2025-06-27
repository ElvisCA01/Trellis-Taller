@echo off
call environment.bat


echo _
echo Current Python: %PYTHON%
echo Virtual Env: %VIRTUAL_ENV%
echo _


rem wrapping in quotation marks in case if path has spaces
cd /d "%~dp0code"


if not exist trellis_init_done.txt (
	
	echo Instalando lo necesario para correr la aplicación. Please wait:
	
	%PYTHON% -m pip install --upgrade pip
    %PYTHON% install.py
	
	if %ERRORLEVEL% == 0 (
		rem If all is good, we will eventually create a file which will know us we ran correctly.
		rem Next time we do this run.bat, the presence of this file will let us skip pip install in this run-gradio.bat
		echo Script ran successfully. Creating init_completed flag file.
		echo initComplete > trellis_init_done.txt
	)
)

rem To catch any issues users might have due to misactivated venv:
echo Current Python: %PYTHON%
echo Virtual Env: %VIRTUAL_ENV%
echo Corriendo el servidor, espera...

rem Notice the %* at the end of  gradio_main.py  will forward the arguments into  gradio_main.py
rem For example  run.bat --precision half    will do  example.py --precision half
%PYTHON% .\gradio_main.py %*

if %ERRORLEVEL% neq 0 (
	echo Something went wrong, consider removing code/trellis_init_done.txt and venv folder to re-initialize from scratch
)

pause

rem Igor Aherne 2025
rem https://github.com/IgorAherne/trellis-stable-projectorz
rem
rem https://stableprojectorz.com
rem Discord: https://discord.gg/aWbnX2qan2