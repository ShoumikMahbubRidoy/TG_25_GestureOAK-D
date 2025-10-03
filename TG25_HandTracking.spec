# TG25_HandTracking.spec
# Build the worker that runs run_hand_tracking.py (no console window)
# Result: dist/TG25_HandTracking/TG25_HandTracking.exe

# Make sure you've installed: pip install pyinstaller pyinstaller-hooks-contrib

block_cipher = None

a = Analysis(
    ['run_hand_tracking.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Keep folder name as 'models' so _find_asset("models/...") works
        ('models', 'models'),
        # Template used by hand_detector.build_manager_script()
        ('src/gesture_oak/utils/template_manager_script_solo.py', 'src/gesture_oak/utils'),
    ],
    hiddenimports=[
        'cv2', 'numpy', 'depthai', 'mediapipe', 'imutils', 'yaml'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='TG25_HandTracking',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
