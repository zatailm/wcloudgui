import sys
import os
from cx_Freeze import setup, Executable
sys.setrecursionlimit(5000)  # Tambahkan ini sebelum kode lainnya

# Menentukan lokasi nltk_data agar tidak hilang dalam build
# os.environ['NLTK_DATA'] = os.path.join(os.path.dirname(__file__), "nltk_data")

# Pastikan lokasi build benar
# Pastikan lokasi DLL
# dll_files = [
#     ("C:/Windows/System32/vcomp140.dll", "vcomp140.dll"),
#     ("C:/Windows/System32/msvcp140.dll", "msvcp140.dll"),
#     ("C:/Windows/System32/vcruntime140.dll", "vcruntime140.dll"),
#     ("C:/Windows/System32/vcruntime140_1.dll", "vcruntime140_1.dll"),
# ]

import shutil

# Lokasi DLL di Windows
dll_list = ["vcomp140.dll", "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"]
dll_files = []

for dll in dll_list:
    src = f"C:/Windows/System32/{dll}"
    dst = os.path.join(os.getcwd(), dll)  # Menyalin ke root folder hasil build
    
    if os.path.exists(src):
        shutil.copy(src, dst)  # Menyalin DLL langsung ke root build
        dll_files.append((dst, dll))  # Menambahkan ke daftar include_files
    else:
        print(f"WARNING: {dll} tidak ditemukan di {src}")

print("DLL files copied:", dll_files)


# Pastikan lokasi transformers
include_files = [
    # ("C:/Users/zata/AppData/Roaming/nltk_data", "lib/nltk_data"),
    # ("D:/miniconda3/envs/wcgen/Lib/site-packages/matplotlib/mpl-data", "lib/matplotlib/mpl-data"),
    ("D:/miniconda3/envs/wcgen/Lib/site-packages/transformers", "lib/transformers"),  # ✅ Tambahkan transformers
] + dll_files  # ✅ Tambahkan DLL

build_exe_options = {
    "packages": [
        "numpy", "matplotlib", "pypdf", "docx",
        "PIL", "flair", "vaderSentiment", "textblob",
        "nltk", "deep_translator", "qasync", "transformers"
    ],
    "includes": [
        "transformers.models.auto",
        "transformers.models.encoder_decoder",
        "transformers.models.bert",
        "transformers.models.roberta",
        "transformers.models.distilbert",
        "transformers.tokenization_utils",
        "transformers.tokenization_utils_base",
    ],
    "excludes": ["tkinter", "PyQt5", "PyQt6", "PySide2"],
    "include_files": include_files,  # ✅ Pastikan daftar file diambil dari include_files yang sudah diperbaiki
}

setup(
    name="WCGen15beta",
    version="1.5b0",
    description="Word Cloud Generator + Sentiment Analysis",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "wcgen15beta.py",
            base="Win32GUI" if sys.platform == "win32" else None,  # GUI Mode
            icon="D:/python_proj/wcloudgui/res/gs.ico"
        )
    ],
)