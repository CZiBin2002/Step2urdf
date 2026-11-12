@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0" || exit /b 1

where /q npm.cmd || (
    echo npm was not found. Install Node.js 22.12 or newer and reopen the terminal.
    goto failed
)

set "VENV_PYTHON=.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" -c "import pip" >nul 2>&1 && goto install
)

set "PYTHON_LAUNCHER=py -3.13"
py -3.13 -c "import venv" >nul 2>&1 || (
    set "PYTHON_LAUNCHER=python"
    python -c "import venv" >nul 2>&1 || (
        echo Python was not found. Install 64-bit Python 3.13 and reopen the terminal.
        goto failed
    )
)

if exist ".venv\" (
    ren ".venv" "venv.bak" || (
        echo Could not back up .venv. Check whether venv.bak already exists.
        goto failed
    )
    echo Moved the unusable .venv to venv.bak.
)
echo Creating the Python environment...
%PYTHON_LAUNCHER% -m venv .venv || goto failed

:install
echo Installing Python dependencies...
"%VENV_PYTHON%" -m pip --disable-pip-version-check install -r requirements.txt || goto failed

if not exist "node_modules\.bin\vite.cmd" goto install_frontend
fc /b "package.json" "node_modules\.step2urdf-package.json" >nul 2>&1 || goto install_frontend
fc /b "package-lock.json" "node_modules\.step2urdf-package-lock.json" >nul 2>&1 && goto build

:install_frontend
echo Installing frontend dependencies...
call npm.cmd ci --include=dev --no-audit --no-fund || goto failed
copy /y "package.json" "node_modules\.step2urdf-package.json" >nul || goto failed
copy /y "package-lock.json" "node_modules\.step2urdf-package-lock.json" >nul || goto failed

:build
echo Building the frontend...
call npm.cmd run build || goto failed

echo Open http://127.0.0.1:8765/ after the server starts. Press Ctrl+C to stop it.
"%VENV_PYTHON%" -m uvicorn workbench.server:app --host 127.0.0.1 --port 8765 || goto failed
popd
exit /b 0

:failed
echo Step2URDF could not start. Check the messages above.
popd
pause
exit /b 1
