# wgen15beta.spec

# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['wcgen15.py'],  # Path to your main Python file
    pathex=['D:/python_proj/wcloudgui/'],  # Path to your project directory
    binaries=[],
    datas=[
        ('D:/miniconda3/Lib/site-packages/matplotlib/mpl-data', 'matplotlib/mpl-data'),
    ],
    hiddenimports=[
        'numpy', 
        'matplotlib',
        'pypdf',
        'docx',
        'PIL',
        'googletrans',
        'flair',
        'vaderSentiment',
        'textblob',
        'matplotlib.backends.backend_ps',
        'matplotlib.backends.backend_pdf',
        'matplotlib.backends.backend_svg',
        'matplotlib.backends.backend_pgf',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WCGen',  # Name of the executable
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=False,  # Set to False for a windowed application
    icon=['D:/python_proj/wcloudgui/res/gs.ico'],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=True,
    upx=True,
    upx_exclude=['arrow.dll', 'torch/lib/*.dll'],
    name='WCGen15beta',  # Name of the final folder
)

# exclude arrow.dll and all dll inside torch/lib from upx compress except torch_cpu.dll (optional: compress it manually with: upx --best --lzma .\lib\torch_cpu.dll)