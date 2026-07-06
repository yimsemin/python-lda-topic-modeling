@echo off
setlocal EnableExtensions EnableDelayedExpansion

if "%~1"=="--help" goto :help
if "%~1"=="-h" goto :help
if "%~1"=="/?" goto :help
if not "%~1"=="" (
    echo [error] Unknown option: %~1
    goto :fail
)

cd /d "%~dp0" || goto :fail

set "PROJECT_ROOT=%CD%"
set "RUNTIME_DIR=%PROJECT_ROOT%\.runtime"
set "RUNTIME_TEMP_DIR=%RUNTIME_DIR%\temp"
set "MATPLOTLIB_CONFIG_DIR=%RUNTIME_DIR%\matplotlib"
set "UV_DIR=%RUNTIME_DIR%\uv"
set "UV_EXE=%UV_DIR%\uv.exe"
set "UV_DOWNLOAD_DIR=%RUNTIME_DIR%\download"
set "UV_CACHE_DIR=%RUNTIME_DIR%\uv-cache"
set "UV_PYTHON_INSTALL_DIR=%RUNTIME_DIR%\python"
set "UV_PROJECT_ENVIRONMENT=%PROJECT_ROOT%\.venv"
set "PYTHON_VERSION_FILE=%PROJECT_ROOT%\.python-version"
set "REQUIREMENTS_FILE=%PROJECT_ROOT%\requirements.txt"
set "PYTHON_EXE=%UV_PROJECT_ENVIRONMENT%\Scripts\python.exe"
set "CURL_EXE=%SystemRoot%\System32\curl.exe"
set "TAR_EXE=%SystemRoot%\System32\tar.exe"
set "TEMP=%RUNTIME_TEMP_DIR%"
set "TMP=%RUNTIME_TEMP_DIR%"

set "UV_STATUS=not checked"
set "PYTHON_STATUS=not checked"
set "VENV_STATUS=not checked"
set "PIP_STATUS=not checked"
set "PACKAGE_STATUS=not checked"
set "FOLDER_STATUS=not checked"

call :load_settings || goto :fail
call :prepare_dirs || goto :fail
call :ensure_uv || goto :fail
call :ensure_python_runtime || goto :fail
call :ensure_venv || goto :fail
call :ensure_pip || goto :fail
call :install_requirements || goto :fail
call :prepare_project_folders || goto :fail

echo.
echo Initial setting completed.
echo Run analysis with:
echo   .venv\Scripts\python.exe run_analysis.py
call :print_summary
goto :finish_success

:load_settings
if not exist "%PYTHON_VERSION_FILE%" (
    echo [error] .python-version was not found.
    exit /b 1
)

for /f "usebackq tokens=* delims=" %%A in ("%PYTHON_VERSION_FILE%") do (
    if not defined PYTHON_VERSION set "PYTHON_VERSION=%%A"
)

if not defined PYTHON_VERSION (
    echo [error] .python-version is empty.
    exit /b 1
)

if not exist "%REQUIREMENTS_FILE%" (
    echo [error] requirements.txt was not found.
    exit /b 1
)

exit /b 0

:prepare_dirs
call :ensure_dir "%RUNTIME_DIR%" || exit /b 1
call :ensure_dir "%RUNTIME_TEMP_DIR%" || exit /b 1
call :ensure_dir "%MATPLOTLIB_CONFIG_DIR%" || exit /b 1
call :ensure_dir "%UV_CACHE_DIR%" || exit /b 1
call :ensure_dir "%UV_PYTHON_INSTALL_DIR%" || exit /b 1
exit /b 0

:ensure_uv
echo [1/6] Checking uv
call :detect_arch || exit /b 1

if not exist "%UV_EXE%" (
    call :download_uv || exit /b 1
    call :get_uv_version UV_VERSION_AFTER
    set "UV_STATUS=installed !UV_VERSION_AFTER!"
    exit /b 0
)

call :get_uv_version UV_VERSION_BEFORE

call :get_latest_uv_version
if errorlevel 1 (
    echo Could not check the latest uv version. Using installed uv.
    set "UV_STATUS=using !UV_VERSION_BEFORE!; latest check unavailable"
    exit /b 0
)

call :compare_versions "!UV_VERSION_BEFORE!" "!UV_LATEST_VERSION!"
if "!VERSION_COMPARE!"=="0" (
    set "UV_STATUS=already up to date: !UV_VERSION_BEFORE!"
    exit /b 0
)

if "!VERSION_COMPARE!"=="1" (
    set "UV_STATUS=using !UV_VERSION_BEFORE!; newer than latest release !UV_LATEST_VERSION!"
    exit /b 0
)

call :download_uv || exit /b 1
call :get_uv_version UV_VERSION_AFTER
set "UV_STATUS=updated from !UV_VERSION_BEFORE! to !UV_VERSION_AFTER!"
exit /b 0

:download_uv
set "UV_DOWNLOAD_URL=https://github.com/astral-sh/uv/releases/latest/download/uv-%UV_ARCH%.zip"
set "UV_ZIP=%UV_DOWNLOAD_DIR%\uv-%UV_ARCH%.zip"
set "UV_EXTRACT_DIR=%UV_DOWNLOAD_DIR%\uv-%UV_ARCH%-%RANDOM%%RANDOM%"

if not exist "%CURL_EXE%" (
    echo [error] curl.exe was not found. Windows 10 or newer is recommended.
    exit /b 1
)

if not exist "%TAR_EXE%" (
    echo [error] tar.exe was not found. Windows 10 or newer is recommended.
    exit /b 1
)

call :ensure_dir "%UV_DIR%" || exit /b 1
call :ensure_dir "%UV_DOWNLOAD_DIR%" || exit /b 1
call :ensure_dir "%UV_EXTRACT_DIR%" || exit /b 1

echo Downloading uv from GitHub
"%CURL_EXE%" --fail --location --retry 3 --output "%UV_ZIP%" "%UV_DOWNLOAD_URL%"
if errorlevel 1 exit /b 1

"%TAR_EXE%" -xf "%UV_ZIP%" -C "%UV_EXTRACT_DIR%"
if errorlevel 1 exit /b 1

set "UV_FOUND="
for /r "%UV_EXTRACT_DIR%" %%F in (uv.exe) do if not defined UV_FOUND (
    copy /y "%%F" "%UV_EXE%" >nul
    set "UV_FOUND=1"
)

set "UVX_FOUND="
for /r "%UV_EXTRACT_DIR%" %%F in (uvx.exe) do if not defined UVX_FOUND (
    copy /y "%%F" "%UV_DIR%\uvx.exe" >nul
    set "UVX_FOUND=1"
)

del /q "%UV_ZIP%" >nul 2>nul
rmdir /s /q "%UV_EXTRACT_DIR%" >nul 2>nul

if not exist "%UV_EXE%" (
    echo [error] uv.exe was not installed.
    exit /b 1
)

exit /b 0

:detect_arch
set "HOST_ARCH=%PROCESSOR_ARCHITECTURE%"
if defined PROCESSOR_ARCHITEW6432 set "HOST_ARCH=%PROCESSOR_ARCHITEW6432%"

set "UV_ARCH="
if /I "%HOST_ARCH%"=="AMD64" set "UV_ARCH=x86_64-pc-windows-msvc"
if /I "%HOST_ARCH%"=="ARM64" set "UV_ARCH=aarch64-pc-windows-msvc"

if not defined UV_ARCH (
    echo [error] Unsupported Windows architecture: %HOST_ARCH%
    exit /b 1
)

exit /b 0

:get_latest_uv_version
set "UV_LATEST_URL="
set "UV_LATEST_VERSION="

if not exist "%CURL_EXE%" exit /b 1

for /f "tokens=* delims=" %%U in ('"%CURL_EXE%" --silent --location --output NUL --write-out "%%{url_effective}" "https://github.com/astral-sh/uv/releases/latest" 2^>nul') do set "UV_LATEST_URL=%%U"

if not defined UV_LATEST_URL exit /b 1

set "UV_LATEST_VERSION=!UV_LATEST_URL:*tag/=!"
if "!UV_LATEST_VERSION!"=="!UV_LATEST_URL!" exit /b 1
if /I "!UV_LATEST_VERSION:~0,1!"=="v" set "UV_LATEST_VERSION=!UV_LATEST_VERSION:~1!"

exit /b 0

:ensure_python_runtime
echo [2/6] Checking Python %PYTHON_VERSION%
call :get_managed_python_version PYTHON_VERSION_BEFORE

"%UV_EXE%" python install "%PYTHON_VERSION%" --managed-python --upgrade --no-bin
if errorlevel 1 (
    if defined PYTHON_VERSION_BEFORE (
        echo [warning] Python update check failed. Continuing with installed Python !PYTHON_VERSION_BEFORE!.
        set "PYTHON_STATUS=using !PYTHON_VERSION_BEFORE!; update check failed"
        exit /b 0
    )
    exit /b 1
)

call :get_managed_python_version PYTHON_VERSION_AFTER
if not defined PYTHON_VERSION_BEFORE (
    set "PYTHON_STATUS=installed !PYTHON_VERSION_AFTER!"
) else if "!PYTHON_VERSION_BEFORE!"=="!PYTHON_VERSION_AFTER!" (
    set "PYTHON_STATUS=already up to date: !PYTHON_VERSION_AFTER!"
) else (
    set "PYTHON_STATUS=updated from !PYTHON_VERSION_BEFORE! to !PYTHON_VERSION_AFTER!"
)
exit /b 0

:ensure_venv
echo [3/6] Checking virtual environment
if exist "%PYTHON_EXE%" (
    call :get_python_exe_version "%PYTHON_EXE%" VENV_VERSION
    set "VENV_STATUS=reused .venv with Python !VENV_VERSION!"
    echo !VENV_VERSION! | findstr /b /c:"%PYTHON_VERSION%" >nul
    if errorlevel 1 (
        echo [warning] Existing .venv uses Python !VENV_VERSION!, but .python-version requests %PYTHON_VERSION%.
        echo [warning] This script does not delete existing environments. Delete .venv and run again if needed.
    )
    exit /b 0
)

if exist "%UV_PROJECT_ENVIRONMENT%" (
    echo Existing .venv is incomplete. Repairing without deleting it.
    "%UV_EXE%" venv --seed --allow-existing --python "%PYTHON_VERSION%" --managed-python "%UV_PROJECT_ENVIRONMENT%"
) else (
    "%UV_EXE%" venv --seed --python "%PYTHON_VERSION%" --managed-python "%UV_PROJECT_ENVIRONMENT%"
)
if errorlevel 1 exit /b 1

call :get_python_exe_version "%PYTHON_EXE%" VENV_VERSION
set "VENV_STATUS=created .venv with Python !VENV_VERSION!"
exit /b 0

:ensure_pip
echo [4/6] Checking pip
"%PYTHON_EXE%" -m pip --version >nul 2>nul
if not errorlevel 1 (
    for /f "tokens=2" %%V in ('"%PYTHON_EXE%" -m pip --version 2^>nul') do set "PIP_VERSION=%%V"
    set "PIP_STATUS=already installed: pip !PIP_VERSION!"
    exit /b 0
)

"%PYTHON_EXE%" -m ensurepip --upgrade
if errorlevel 1 exit /b 1

for /f "tokens=2" %%V in ('"%PYTHON_EXE%" -m pip --version 2^>nul') do set "PIP_VERSION=%%V"
set "PIP_STATUS=installed pip !PIP_VERSION!"
exit /b 0

:install_requirements
echo [5/6] Checking packages from requirements.txt
"%UV_EXE%" pip install --strict --python "%PYTHON_EXE%" -r "%REQUIREMENTS_FILE%"
if errorlevel 1 exit /b 1

set "PACKAGE_STATUS=requirements.txt satisfied; uv output above shows any installs or updates"
exit /b 0

:prepare_project_folders
echo [6/6] Preparing project folders
call :ensure_dir "%PROJECT_ROOT%\input" || exit /b 1
call :ensure_dir "%PROJECT_ROOT%\output" || exit /b 1
set "FOLDER_STATUS=input and output folders are ready"
exit /b 0

:print_summary
echo.
echo Summary:
echo   uv: %UV_STATUS%
echo   Python runtime: %PYTHON_STATUS%
echo   Virtual environment: %VENV_STATUS%
echo   pip: %PIP_STATUS%
echo   Packages: %PACKAGE_STATUS%
echo   Project folders: %FOLDER_STATUS%
echo.
echo Requirements:
for /f "usebackq tokens=* delims=" %%R in ("%REQUIREMENTS_FILE%") do (
    if not "%%R"=="" echo   %%R
)
exit /b 0

:get_uv_version
set "%~1="
if exist "%UV_EXE%" (
    for /f "tokens=2" %%V in ('"%UV_EXE%" self version 2^>nul') do set "%~1=%%V"
)
if not defined %~1 set "%~1=unknown"
exit /b 0

:get_managed_python_version
set "%~1="
for /d %%D in ("%UV_PYTHON_INSTALL_DIR%\cpython-%PYTHON_VERSION%*-windows-*-none") do (
    if exist "%%~fD\python.exe" call :get_python_exe_version "%%~fD\python.exe" %~1
)
exit /b 0

:compare_versions
set "VERSION_COMPARE=0"

set "A1=0"
set "A2=0"
set "A3=0"
set "B1=0"
set "B2=0"
set "B3=0"

for /f "tokens=1-3 delims=.vV-" %%A in ("%~1") do (
    if not "%%A"=="" set "A1=%%A"
    if not "%%B"=="" set "A2=%%B"
    if not "%%C"=="" set "A3=%%C"
)

for /f "tokens=1-3 delims=.vV-" %%A in ("%~2") do (
    if not "%%A"=="" set "B1=%%A"
    if not "%%B"=="" set "B2=%%B"
    if not "%%C"=="" set "B3=%%C"
)

if !A1! LSS !B1! set "VERSION_COMPARE=-1" & exit /b 0
if !A1! GTR !B1! set "VERSION_COMPARE=1" & exit /b 0
if !A2! LSS !B2! set "VERSION_COMPARE=-1" & exit /b 0
if !A2! GTR !B2! set "VERSION_COMPARE=1" & exit /b 0
if !A3! LSS !B3! set "VERSION_COMPARE=-1" & exit /b 0
if !A3! GTR !B3! set "VERSION_COMPARE=1" & exit /b 0
exit /b 0

:get_python_exe_version
set "%~2="
if exist "%~1" (
    for /f "tokens=2" %%V in ('"%~1" --version 2^>nul') do set "%~2=%%V"
)
if not defined %~2 set "%~2=unknown"
exit /b 0

:ensure_dir
if not exist "%~1" mkdir "%~1"
exit /b %ERRORLEVEL%

:help
echo Usage: initial_setting.bat
echo.
echo Creates or updates this project's local Windows runtime.
echo - Missing uv, Python, pip, and packages are installed.
echo - Existing uv and managed Python are updated when a newer version is available.
echo - Existing .venv is reused; it is never deleted by this script.
echo - Package versions are adjusted to satisfy requirements.txt.
echo - No administrator permission or global PATH change is required.
goto :finish_success

:fail
echo.
echo Initial setting failed.
call :print_summary
goto :finish_fail

:finish_success
echo.
pause
exit /b 0

:finish_fail
echo.
pause
exit /b 1
