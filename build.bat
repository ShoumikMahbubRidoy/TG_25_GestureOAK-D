@echo off
setlocal
REM Clean
rmdir /s /q build dist 2>nul

python -m pip install --upgrade pip pyinstaller pyinstaller-hooks-contrib

REM 1) Build worker (packs models + depthai)
pyinstaller --noconfirm --clean TG25_HandTracking.spec
IF ERRORLEVEL 1 ( echo Worker build failed & exit /b 1 )

REM 2) Build launcher (bundles the worker exe next to it)
pyinstaller --noconfirm --clean TG25_Launcher.spec
IF ERRORLEVEL 1 ( echo Launcher build failed & exit /b 1 )

echo.
echo Build complete. Run: dist\TG25_Launcher\TG25_Launcher.exe
