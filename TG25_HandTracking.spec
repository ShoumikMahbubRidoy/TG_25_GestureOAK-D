# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['run_hand_tracking.py'],
    pathex=[],
    binaries=[],
    datas=[('models', 'pyd_models'), ('src/gesture_oak/utils/template_manager_script_solo.py', 'src/gesture_oak/utils')],
    hiddenimports=['depthai', 'cv2', 'numpy'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

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
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
