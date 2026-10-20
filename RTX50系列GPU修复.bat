@echo off
rem GPU repair for RTX 50-series (Blackwell) GPUs: switch torch to the
rem cu128 build (torch 2.7.1+cu128), which has sm_120 kernels the default
rem cu124 build lacks. Kept ASCII-only on purpose (see the other two bats).
setlocal
set "ROOT=%~dp0"
if not exist "%ROOT%python_env\python.exe" (
  echo [ERROR] python_env\python.exe not found under %ROOT%
  echo This file must stay inside the MIMICS_SCRIPTS folder.
  pause
  exit /b 1
)
if not exist "%ROOT%repair\cu118\switch_gpu_torch.py" (
  echo [ERROR] repair\cu118\switch_gpu_torch.py not found under %ROOT%
  echo The repair kit is missing; please re-copy the MIMICS_SCRIPTS folder.
  pause
  exit /b 1
)
"%ROOT%python_env\python.exe" "%ROOT%repair\cu118\switch_gpu_torch.py" switch128
echo.
pause
