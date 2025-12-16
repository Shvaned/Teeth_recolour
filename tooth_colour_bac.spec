# -*- mode: python ; coding: utf-8 -*-
import os
import PyQt5
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

block_cipher = None

opencv_binaries = collect_dynamic_libs('cv2')
pyqt5_hiddenimports = collect_submodules('PyQt5')

hookspath = []

datas = [
    ('mouth_nano.pt', '.'),
    ('tooth_nano.pt', '.'),
    ('idle.png', '.'),
    ('smile.png', '.'),
    ('promo.png', '.'),
]

a = Analysis(
    ['tooth_colour.py'],
    pathex=[],
    binaries=opencv_binaries,
    datas=datas,
    hiddenimports=[
        'cv2',
        'torch',            # keep only the base
        'torchvision',      # if you’re actually using it
        'ultralytics',
        'PIL',
        'PIL._imaging',
        *pyqt5_hiddenimports
    ],
    hookspath=hookspath,
    excludes=[
        # Drop entire distributed & RPC & CUDA backends
        'torch.distributed', 'torch.distributed.*',
        'torch._inductor',   'torch._inductor.*',
        'torch._dynamo',     'torch._dynamo.*',
        'torch._functorch',  'torch._functorch.*',
        'torch.fx',          'torch.fx.*',
        'torch.backends',    'torch.backends.*',
        'torch.cuda',        'torch.cuda.*',
        'torch.jit',         'torch.jit.*',
        'torch.autograd',    'torch.autograd.*',
        'torch.optim',       'torch.optim.*',
        'torch.testing',     'torch.testing.*',
        'torch.utils',       'torch.utils.*',
    ],
    runtime_hooks=[],
    noarchive=True,       # don’t pack pure-Python modules into base_library.zip
    optimize=0
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name='tooth_colour',
    debug=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='tooth_colour'
)
