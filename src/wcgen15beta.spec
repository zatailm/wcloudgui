# wgen15beta.spec

# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['WCGen15beta.py'],                         # Path ke script utama
    pathex=['D:/python_proj/wcloudgui/'],       # Path ke proyek
    binaries=[],
    datas=[
        ('D:/miniconda3/envs/wcgen/Lib/site-packages/matplotlib/mpl-data', 'matplotlib/mpl-data'),
        ('C:/Users/zata/AppData/Roaming/nltk_data', 'nltk_data'),  # Pastikan seluruh nltk_data disertakan
    ],
    hiddenimports=[
        'numpy', 
        'matplotlib',
        'pypdf',
        'docx',
        'PIL',
        'flair',
        'vaderSentiment',
        'textblob',
        'nltk',
        'nltk.corpus',
        'nltk.tokenize',
        'nltk.stem',
        'nltk.tag',
        'nltk.data',
        'nltk.chunk',
        'nltk.parse',
        'deep_translator',
        'qasync',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PyQt5',    # Exclude PyQt5
        'PyQt6',    # Exclude PyQt6
        'PySide2',  # Exclude PySide2
        'tkinter',  # Exclude tkinter jika tidak digunakan
    ],
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
    name='WCGen15beta',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,  # Mengurangi ukuran .exe
    upx=True,
    console=False,  # False jika aplikasi GUI, True jika ingin melihat output terminal
    icon=['D:/python_proj/wcloudgui/res/gs.ico'],
    onefile=True,
)

# Hapus/nonaktifkan bagian bawah jika onefile=True
# coll = COLLECT(
#     exe,
#     a.binaries,
#     a.zipfiles,
#     a.datas,
#     strip=True,
#     upx=True,
#     upx_exclude=[
#         'arrow.dll', 
#         'torch/lib/*.dll',  
#         'numpy/core/*.dll',  
#         'scipy/*.dll',  
#         'matplotlib/*.dll',  
#         'libopenblas*.dll',  
#         'libgfortran*.dll',  
#         'libquadmath*.dll',  
#         'vcruntime140.dll',  
#         'msvcp140.dll',  
#     ],
#     name='WCGen15beta',
# )

# exclude arrow.dll and all dll inside torch/lib from upx compress except torch_cpu.dll (optional: compress it manually with: upx --best --lzma .\lib\torch_cpu.dll)
# pyinstaller --noconfirm --onedir --windowed --icon "D:\python_proj\wcloudgui\res\gs.ico" --name "WCGen1.5b0" --upx-dir "C:\upx" --clean --optimize "2" --strip --add-data "D:\miniconda3\envs\wcgen\Lib\site-packages\transformers;transformers/" --exclude-module "tkinter" --exclude-module "PyQt5" --exclude-module "PyQt6" --exclude-module "PySide2" --upx-exclude "arrow.dll" --upx-exclude "torch/lib/*.dll" --hidden-import "numpy" --hidden-import "matplotlib" --hidden-import "pypdf" --hidden-import "docx" --hidden-import "PIL" --hidden-import "flair" --hidden-import "vaderSentiment" --hidden-import "textblob" --hidden-import "deep_translator" --hidden-import "qasync"  "D:\python_proj\wcloudgui\src\wcgen15beta.py"