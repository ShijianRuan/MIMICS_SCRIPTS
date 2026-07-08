@echo off

setlocal enabledelayedexpansion

echo ========================================

echo   Mimics-Script Offline Setup

echo   Target: Windows + Python 3.13.7

echo ========================================

echo.



cd /d %~dp0



:: 1. Setup Python embeddable (self-contained, no system Python needed)

echo [1/5] Setting up Python 3.13.7...

if not exist "nninteractive_env\python.exe" (

    echo   Extracting Python embeddable...

    if not exist "python\python.exe" (

        echo   ERROR: python\python.exe not found.

        echo   Make sure the python/ directory is present.

        pause

        exit /b 1

    )

    mkdir nninteractive_env 2>nul

    xcopy /E /Q /Y python\* nninteractive_env\ >nul

    echo   Configuring python313._pth...

    echo python313.zip > nninteractive_env\python313._pth

    echo . >> nninteractive_env\python313._pth

    echo Lib\site-packages >> nninteractive_env\python313._pth

    echo import site >> nninteractive_env\python313._pth

)

if not exist "nninteractive_env\Lib\site-packages" mkdir nninteractive_env\Lib\site-packages



:: 2. Install pip (try ensurepip, fallback to get-pip.py)

echo [2/5] Installing pip...

nninteractive_env\python.exe -m pip --version >nul 2>&1

if !errorlevel! neq 0 (

    echo   Trying ensurepip...

    nninteractive_env\python.exe -m ensurepip --upgrade 2>nul

    nninteractive_env\python.exe -m pip --version >nul 2>&1

    if !errorlevel! neq 0 (

        echo   ensurepip failed. Trying get-pip.py...

        if exist "get-pip.py" (

            nninteractive_env\python.exe get-pip.py --no-index --find-links="wheels" pip setuptools wheel 2>nul

            nninteractive_env\python.exe -m pip --version >nul 2>&1

            if !errorlevel! neq 0 (

                echo   ERROR: pip installation failed.

                echo   Manual: nninteractive_env\python.exe get-pip.py --no-index --find-links="wheels" pip setuptools wheel

                pause

                exit /b 1

            )

        ) else (

            echo   ERROR: get-pip.py not found and ensurepip failed.

            echo   pip is required to install packages.

            pause

            exit /b 1

        )

    )

)

echo   pip is ready.



:: 3. Check package directory

echo [3/5] Checking wheels...

if not exist "wheels\*.whl" if not exist "wheels\*.tar.gz" (

    echo   ERROR: No package files found in wheels\ directory.

    echo   The wheels/ directory must contain .whl and/or .tar.gz packages.

    pause

    exit /b 1

)

dir /b wheels\*.whl ^| find /c /v "" >nul 2>&1

echo   Wheels directory OK.



:: 4. Install all packages offline (no internet, no system Python)

echo [4/5] Installing packages from local wheels (no internet)...

set FAIL_COUNT=0

for %%f in (wheels\*.whl) do (

    echo   Installing %%~nxf...

    nninteractive_env\python.exe -m pip install "%%f" --no-deps --no-index --quiet 2>nul

    if !errorlevel! neq 0 (

        echo     WARNING: Failed to install %%~nxf

        set /a FAIL_COUNT+=1

    )

)

for %%f in (wheels\*.tar.gz) do (

    echo   Installing %%~nxf...

    nninteractive_env\python.exe -m pip install "%%f" --no-deps --no-index --no-build-isolation --quiet 2>nul

    if !errorlevel! neq 0 (

        echo     WARNING: Failed to install %%~nxf

        set /a FAIL_COUNT+=1

    )

)

echo   Installation complete. !FAIL_COUNT! package(s) failed.



:: 5. Verify

echo [5/5] Verifying installation...

nninteractive_env\python.exe -c "import torch; print('  torch', torch.__version__); print('  CUDA available:', torch.cuda.is_available())"

if !errorlevel! neq 0 (

    echo   ERROR: torch import failed.

    pause

    exit /b 1

)

nninteractive_env\python.exe -c "import numpy, nibabel, pydicom, SimpleITK, scipy, nnInteractive, torchvision, transformers, yaml, tqdm, acvl_utils; print('  All packages OK')"

if !errorlevel! neq 0 (

    echo   Some packages failed to import.

    echo   Try: nninteractive_env\python.exe -m pip install wheels\*.whl --no-deps --no-index
    echo   And: nninteractive_env\python.exe -m pip install wheels\*.tar.gz --no-deps --no-index --no-build-isolation

    pause

    exit /b 1

)



echo.

echo ========================================

echo   Setup complete!

echo   In Mimics: Scripting -^> Add Scripting Library

echo   Select: scripting_library\ folder

echo ========================================

pause