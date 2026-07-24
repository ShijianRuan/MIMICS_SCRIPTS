@echo off

setlocal enabledelayedexpansion

echo ========================================

echo   Mimics-Script Offline Setup

echo   Target: Windows + Python 3.13.7

echo ========================================

echo.



cd /d %~dp0



:: 1. Setup Python embeddable (self-contained, no system Python needed)

echo [1/6] Setting up Python 3.13.7...

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

echo [2/6] Installing pip...

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

echo [3/6] Checking wheels...

if not exist "wheels\*.whl" if not exist "wheels\*.tar.gz" (

    echo   ERROR: No package files found in wheels\ directory.

    echo   The wheels/ directory must contain .whl and/or .tar.gz packages.

    pause

    exit /b 1

)

dir /b wheels\*.whl ^| find /c /v "" >nul 2>&1

echo   Wheels directory OK.



:: 4. Install all packages offline (no internet, no system Python)

echo [4/6] Installing packages from local wheels (no internet)...

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



:: 5. Install tkinter (embedded Python does not ship with it)

echo [5/6] Installing tkinter support...

if exist "tkinter_support\DLLs\_tkinter.pyd" (

    copy /Y "tkinter_support\DLLs\_tkinter.pyd" "nninteractive_env\" >nul

    copy /Y "tkinter_support\DLLs\tcl86t.dll" "nninteractive_env\" >nul

    copy /Y "tkinter_support\DLLs\tk86t.dll" "nninteractive_env\" >nul

    copy /Y "tkinter_support\DLLs\zlib1.dll" "nninteractive_env\" >nul

    if not exist "nninteractive_env\Lib\tkinter" (

        xcopy /E /Q /Y "tkinter_support\Lib\tkinter" "nninteractive_env\Lib\tkinter\" >nul

    )

    if not exist "nninteractive_env\tcl\tk8.6" (

        xcopy /E /Q /Y "tkinter_support\tcl" "nninteractive_env\tcl\" >nul

    )

    nninteractive_env\python.exe -c "import tkinter; print('  tkinter OK')" 2>nul

    if !errorlevel! neq 0 (

        echo   WARNING: tkinter installation failed. External UI will fall back to profile selector.

    ) else (

        echo   tkinter is ready.

    )

) else (

    echo   tkinter_support not found. External UI will fall back to profile selector.

)



:: 6. Verify

echo [6/6] Verifying installation...

nninteractive_env\python.exe -c "import torch; print('  torch', torch.__version__); print('  CUDA available:', torch.cuda.is_available())"

if !errorlevel! neq 0 (

    echo   ERROR: torch import failed.

    pause

    exit /b 1

)

nninteractive_env\python.exe -c "import numpy, nibabel, pydicom, SimpleITK, scipy, nnInteractive, torchvision, transformers, yaml, tqdm, acvl_utils, onnxruntime; print('  All packages OK'); print('  ONNX providers:', ', '.join(onnxruntime.get_available_providers()))"

if !errorlevel! neq 0 (

    echo   Some packages failed to import. The default frozen-feature method requires onnxruntime-gpu.

    echo   Try: nninteractive_env\python.exe -m pip install wheels\*.whl --no-deps --no-index
    echo   And: nninteractive_env\python.exe -m pip install wheels\*.tar.gz --no-deps --no-index --no-build-isolation

    pause

    exit /b 1

)

if not exist "external\dinov3-medical-seg\models\dinov3-vits16\model.onnx" (

    echo   ERROR: The default ViT-S/16 ONNX encoder is missing.

    echo   Expected: external\dinov3-medical-seg\models\dinov3-vits16\model.onnx

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
