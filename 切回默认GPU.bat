@echo off
rem GPU restore: undo the cu118 switch (back to the default cu124 build).
rem ASCII-only on purpose: cmd.exe parses .bat files in the legacy OEM/ANSI
rem codepage, so non-ASCII text here risks mojibake on other machines. All
rem user-facing (Chinese) output comes from the Python script.
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
"%ROOT%python_env\python.exe" "%ROOT%repair\cu118\switch_gpu_torch.py" restore
echo.
pause
