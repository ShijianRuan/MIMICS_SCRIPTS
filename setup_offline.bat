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
if not exist "python_env\python.exe" (
    echo   Extracting Python embeddable...
    if not exist "python\python.exe" (
        echo   ERROR: python\python.exe not found.
        echo   Make sure the python/ directory is present.
        pause
        exit /b 1
    )
    mkdir python_env 2>nul
    xcopy /E /Q /Y python\* python_env\ >nul
    echo   Configuring python313._pth...
    echo python313.zip > python_env\python313._pth
    echo . >> python_env\python313._pth
    echo Lib >> python_env\python313._pth
    echo Lib\site-packages >> python_env\python313._pth
    echo import site >> python_env\python313._pth
)
:: Ensure Lib is on the path (idempotent; also covers upgraded installs)
echo python313.zip > python_env\python313._pth
echo . >> python_env\python313._pth
echo Lib >> python_env\python313._pth
echo Lib\site-packages >> python_env\python313._pth
echo import site >> python_env\python313._pth
if not exist "python_env\Lib\site-packages" mkdir python_env\Lib\site-packages

:: 2. Install pip (try ensurepip, fallback to get-pip.py)
echo [2/6] Installing pip...
python_env\python.exe -m pip --version >nul 2>&1
if !errorlevel! neq 0 (
    echo   Trying ensurepip...
    python_env\python.exe -m ensurepip --upgrade 2>nul
    python_env\python.exe -m pip --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo   ensurepip failed. Trying get-pip.py...
        if exist "get-pip.py" (
            python_env\python.exe get-pip.py --no-index --find-links="wheels" pip setuptools wheel 2>nul
            python_env\python.exe -m pip --version >nul 2>&1
            if !errorlevel! neq 0 (
                echo   ERROR: pip installation failed.
                echo   Manual: python_env\python.exe get-pip.py --no-index --find-links="wheels" pip setuptools wheel
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

:: 3. Check wheels directory
echo [3/6] Checking wheels...
if not exist "wheels\*.whl" (
    echo   ERROR: No .whl files found in wheels\ directory.
    echo   The wheels/ directory must contain all required packages.
    pause
    exit /b 1
)
dir /b wheels\*.whl ^| find /c /v "" >nul 2>&1
echo   Wheels directory OK.

:: 4. Install all wheels offline (no internet, no system Python)
echo [4/6] Installing packages from local wheels (no internet)...
set FAIL_COUNT=0
for %%f in (wheels\*.whl) do (
    echo   Installing %%~nxf...
    python_env\python.exe -m pip install "%%f" --no-deps --no-index --quiet 2>nul
    if !errorlevel! neq 0 (
        echo     WARNING: Failed to install %%~nxf
        set /a FAIL_COUNT+=1
    )
)
echo   Installation complete. !FAIL_COUNT! package(s) failed.
echo   Ensuring PySide6 advanced UI wheels are installed consistently...
python_env\python.exe -m pip install PySide6 shiboken6 --no-index --find-links="wheels" --upgrade --quiet
if !errorlevel! neq 0 (
    echo   ERROR: PySide6 installation failed.
    echo   Make sure wheels\ contains matching PySide6, PySide6_Essentials, PySide6_Addons, and shiboken6 Windows wheels.
    pause
    exit /b 1
)

:: 5. Verify external GUI backend
echo [5/6] Verifying PySide6 external UI backend...
python_env\python.exe -c "import PySide6, shiboken6; from PySide6 import QtCore, QtWidgets; print('  PySide6', QtCore.__version__)"
if !errorlevel! neq 0 (
    echo   ERROR: PySide6 import failed.
    echo   Advanced DINOv3 Setup and Status windows require PySide6 in python_env.
    pause
    exit /b 1
)

:: 6. Verify
echo [6/6] Verifying installation...
python_env\python.exe -c "import torch; print('  torch', torch.__version__); print('  CUDA available:', torch.cuda.is_available())"
if !errorlevel! neq 0 (
    echo   ERROR: torch import failed.
    pause
    exit /b 1
)
python_env\python.exe -c "import numpy, nibabel, pydicom, SimpleITK, scipy, nnInteractive, nnunetv2, torchvision, transformers, yaml, tqdm, tensorboard, tomli, acvl_utils, onnxruntime, PySide6, shiboken6; from importlib.metadata import version as package_version; from packaging.version import Version; nnv=Version(package_version('nnunetv2')); assert Version('2.8.1') ^<= nnv ^< Version('2.9'), 'nnunetv2 2.8.1 through 2.8.x is required'; print('  All packages OK'); print('  nnU-Net', nnv); print('  ONNX providers:', ', '.join(onnxruntime.get_available_providers()))"
if !errorlevel! neq 0 (
    echo   Some packages failed to import. The default frozen-feature method requires onnxruntime-gpu.
    echo   Try: python_env\python.exe -m pip install wheels\*.whl --no-deps --no-index
    pause
    exit /b 1
)
python_env\python.exe -c "import paramiko; print('  Optional remote training transport ready:', paramiko.__version__)"
if !errorlevel! neq 0 (
    echo   WARNING: Paramiko is unavailable. Local training is unaffected; remote training is disabled.
)
if not exist "external\dinov3-medical-seg\models\dinov3-vits16\model.onnx" (
    echo   ERROR: The default ViT-S/16 ONNX encoder is missing.
    echo   Expected: external\dinov3-medical-seg\models\dinov3-vits16\model.onnx
    pause
    exit /b 1
)
if not exist "external\ScribblePrompt\checkpoints\ScribblePrompt_unet_v1_nf192_res128.pt" (
    echo   WARNING: The official ScribblePrompt UNet checkpoint is missing.
    echo   Expected: external\ScribblePrompt\checkpoints\ScribblePrompt_unet_v1_nf192_res128.pt
)

echo.
echo ========================================
echo   Setup complete!
echo   In Mimics: Scripting -^> Add Scripting Library
echo   Select: scripting_library\ folder
echo ========================================
pause