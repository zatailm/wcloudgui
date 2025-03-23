import sys
import os
import shutil
import platform
from cx_Freeze import setup, Executable
import vaderSentiment
import wordcloud
import transformers

sys.setrecursionlimit(5000)

# Get paths
vader_lexicon_path = os.path.join(os.path.dirname(vaderSentiment.__file__), "vader_lexicon.txt")
wordcloud_path = os.path.dirname(wordcloud.__file__)
stopwords_path = os.path.join(wordcloud_path, "stopwords")

# DLL handling with multiple possible locations
base_dlls = ["vcomp140.dll", "msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"]
additional_dlls = ["zlib.dll"]
dll_files = []
root_dlls = []  # For DLLs that should go to root folder

def get_python_dll_path():
    if hasattr(sys, 'base_prefix'):
        return os.path.join(sys.base_prefix, 'DLLs')
    return os.path.join(sys.exec_prefix, 'DLLs')

# Extended search paths
dll_search_paths = [
    "C:/Windows/System32",
    "C:/Windows/SysWOW64",
    get_python_dll_path(),
    os.path.dirname(sys.executable),
    # Additional paths for zlib.dll
    "C:/Program Files/Python39",
    "C:/Program Files/Python310",
    "C:/Program Files/Python311",
    "C:/Program Files (x86)/Python39",
    "C:/Program Files (x86)/Python310", 
    "C:/Program Files (x86)/Python311",
    "D:/miniconda3/envs/wcgen/Library/bin",
    # Add Visual Studio paths
]

# Visual Studio 2015-2022 paths
vs_years = range(2015, 2023)
for year in vs_years:
    dll_search_paths.extend([
        f"C:/Program Files (x86)/Microsoft Visual Studio/{year}/Community/VC/Redist/MSVC/14.0/x64/Microsoft.VC140.CRT",
        f"C:/Program Files (x86)/Microsoft Visual Studio/{year}/Community/VC/Redist/MSVC/14.0/x86/Microsoft.VC140.CRT",
        f"C:/Program Files (x86)/Microsoft Visual Studio/{year}/BuildTools/VC/Redist/MSVC/14.0/x64/Microsoft.VC140.CRT",
        f"C:/Program Files (x86)/Microsoft Visual Studio/{year}/BuildTools/VC/Redist/MSVC/14.0/x86/Microsoft.VC140.CRT",
    ])

# Try to find and copy base DLLs first
for dll in base_dlls:
    dll_found = False
    for search_path in dll_search_paths:
        if os.path.exists(search_path):
            src = os.path.join(search_path, dll)
            if os.path.exists(src):
                dst = os.path.join(os.getcwd(), "dlls", dll)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    shutil.copy2(src, dst)
                    dll_files.append((dst, os.path.join("lib", dll)))
                    print(f"Successfully copied {dll} from {src}")
                    dll_found = True
                    break
                except Exception as e:
                    print(f"Failed to copy {dll} from {src}: {e}")
    
    if not dll_found:
        print(f"WARNING: {dll} not found in any search path")

# Try to find and copy additional DLLs with more paths
for dll in additional_dlls:
    dll_found = False
    # Add conda/python specific paths for zlib
    extra_paths = [
        os.path.join(sys.prefix, 'Library', 'bin'),
        os.path.join(sys.prefix, 'DLLs'),
        os.path.dirname(sys.executable),
    ] + dll_search_paths
    
    for search_path in extra_paths:
        if os.path.exists(search_path):
            src = os.path.join(search_path, dll)
            if os.path.exists(src):
                dst = os.path.join(os.getcwd(), "dlls", dll)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    shutil.copy2(src, dst)
                    if dll == "zlib.dll":
                        root_dlls.append((dst, dll))  # Copy to root
                    else:
                        dll_files.append((dst, os.path.join("lib", dll)))
                    print(f"Successfully copied {dll} from {src}")
                    dll_found = True
                    break
                except Exception as e:
                    print(f"Failed to copy {dll} from {src}: {e}")
    
    if not dll_found:
        print(f"WARNING: {dll} not found in any search path - application may still work without it")

# Prepare wordcloud resources and packages
wordcloud_dir = os.path.dirname(wordcloud.__file__)
wordcloud_resources = [
    (os.path.join(wordcloud_dir, "stopwords"), "wordcloud/stopwords"),  # Pindah ke root folder
    (wordcloud_dir, "wordcloud"),  # Include seluruh package wordcloud
]

# Prepare transformers resources
transformers_dir = os.path.dirname(transformers.__file__)
transformers_resources = [
    (transformers_dir, "transformers"),  # Include entire transformers package
]

# Get package directories
package_dirs = {
    "transformers": transformers_dir,
    "wordcloud": wordcloud_dir,
    "vaderSentiment": os.path.dirname(vaderSentiment.__file__),
    "textblob": os.path.join(os.path.dirname(sys.executable), "Lib/site-packages/textblob"),
    "flair": os.path.join(os.path.dirname(sys.executable), "Lib/site-packages/flair"),
    "nltk": os.path.join(os.path.dirname(sys.executable), "Lib/site-packages/nltk"),
}

# Package resources
package_resources = []
for pkg_name, pkg_dir in package_dirs.items():
    if os.path.exists(pkg_dir):
        package_resources.append((pkg_dir, f"lib/{pkg_name}"))
    else:
        print(f"WARNING: Package directory not found: {pkg_dir}")

# Define include files
include_files = [
    ("D:/pythonpro/wcloudgui/res/LICENSE.txt", "LICENSE.txt"),
    ("D:/pythonpro/wcloudgui/src/icon.ico", "icon.ico"),
] + package_resources + dll_files + root_dlls  # Add root_dlls separately

# Verify files exist
for src, dst in include_files:
    if not os.path.exists(src):
        print(f"WARNING: {src} tidak ditemukan! File ini akan dilewati.")

build_exe_options = {
    "packages": [
        "numpy", "matplotlib", "pypdf", "docx",
        "PIL", "deep_translator", "qasync",
    ],
    "includes": [
        "transformers",
        "transformers.models",
        "transformers.models.auto",
        "transformers.models.encoder_decoder",
        "transformers.models.bert",
        "transformers.models.roberta",
        "transformers.models.distilbert",
        "transformers.models.albert",
        "transformers.modeling_utils",
        "transformers.tokenization_utils",
        "transformers.tokenization_utils_base",
    ],
    "excludes": ["tkinter", "PyQt5", "PyQt6", "PySide2", "torchaudio", "torchvision"],
    "include_files": include_files,
    "zip_include_packages": [],  # Don't zip any packages
    "zip_exclude_packages": ["*"],  # Exclude all packages from zip
    "build_exe": "build/exe.win-amd64-3.12",
    "include_msvcr": True,
}

setup(
    name="WCGen1.5",
    version="1.5",
    description="Word Cloud Generator + Sentiment Analysis",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "wcgen15.py",
            base="Win32GUI" if sys.platform == "win32" else None,
            icon="D:/pythonpro/wcloudgui/src/icon.ico"
        )
    ],
)
