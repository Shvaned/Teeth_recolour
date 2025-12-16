# tooth_recolour.spec

# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

a = Analysis(
    ['tooth_colour.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('mouth_nano.pt', '.'),
        ('tooth_nano.pt', '.'),
        ('idle.png', '.'),
        ('smile.png', '.'),
        ('promo.png', '.'),
    ],
    hiddenimports=[
        'cv2',
        'torch',
        'ultralytics',
        'torch.distributed',
        *collect_submodules('ultralytics'),  # Optional, if needed
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
    'tkinter',
    'tensorflow',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='tooth_recolour',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='tooth_recolour',
)
